import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { SessionManager, EventLogger } from '@/lib/eventLogger';
import { getChurnPrediction, getActiveUserState, BackendDisconnectedError, ChurnBreakdownItem } from '@/lib/fastApiClient';
import { AlertCircle, TrendingDown, ChevronRight, ChevronLeft } from 'lucide-react';

const MAX_POINTS = 40;   // Churn Rate 추이 선그래프에 보관할 최근 포인트 수
// 활동 전(세션 이벤트 0) 표시용 0% breakdown — 초기값이 박혀 보이지 않게
const ZERO_BREAKDOWN: ChurnBreakdownItem[] = [
  { key: 'churn_7d', label: '7일 이탈(집계모델)', probability: 0 },
  { key: 'hazard', label: '실시간 하자드', probability: 0 },
  { key: 'bounce', label: '이탈 Bounce(30분)', probability: 0 },
];

export default function FloatingChurnWidget() {
  const [churnRate, setChurnRate] = useState<number | null>(null);
  const [breakdown, setBreakdown] = useState<ChurnBreakdownItem[]>([]);
  const [history, setHistory] = useState<number[]>([]);   // Churn Rate 시계열(선그래프)
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [refreshIntervalSec, setRefreshIntervalSec] = useState(4);
  const session = SessionManager.getOrCreateSession();

  useEffect(() => {
    let active = true;
    // 세션 이벤트를 백엔드(/api/churn/predict)로 보내 실제 모델 이탈확률을 받는다(랜덤 아님).
    const fetchChurnRate = async () => {
      setIsLoading(true);
      try {
        const raw = await EventLogger.getSessionEvents(session.sessionId);
        const events = (raw as Record<string, unknown>[]).map(e => ({
          event_type: String(e.eventType ?? e.event_type ?? 'view'),
          product_id: String(e.productId ?? e.product_id ?? ''),
          category_id: String(e.categoryId ?? e.category_id ?? ''),
          brand: String(e.brand ?? ''),
          price: Number(e.price ?? 0),
          quantity: Number(e.quantity ?? 1),
          // 백엔드 hazard 가 inter-event 간격/유휴를 쓰므로 반드시 ISO 로 보낸다.
          timestamp: e.eventTime instanceof Date
            ? e.eventTime.toISOString()
            : String(e.eventTime ?? e.event_time ?? new Date().toISOString()),
        }));
        // 활동 전(이벤트 0) = 초기값. 백엔드 baseline(bounce 등) 대신 0%로 표시(사용자 요청).
        if (events.length === 0) {
          if (active) {
            setChurnRate(0);
            setBreakdown(ZERO_BREAKDOWN);
            setHistory(h => [...h, 0].slice(-MAX_POINTS));
            setError(null);
          }
          return;
        }
        // 무상태 채점: 호출마다 ephemeral session_id → 백엔드가 버퍼 전체를 1회만 채점(이중 누적 방지).
        const churnSid = `${session.sessionId}:t${Date.now()}`;
        // 대시보드가 선택한 유저(active-user)가 있으면 그 유저로 채점 귀속 → 대시보드 실시간이 살아 움직임.
        const activeState = await getActiveUserState();
        const nextInterval = Math.max(1, Math.min(60, Number(activeState.refresh_interval_sec) || 4));
        if (active && nextInterval !== refreshIntervalSec) {
          setRefreshIntervalSec(nextInterval);
        }
        const resp = await getChurnPrediction(churnSid, activeState.user_id || session.userId, events);
        if (!active) return;
        const bd = resp.churn_breakdown ?? [];
        setBreakdown(bd);
        // 헤드라인 = 서버가 대시보드 정책(max/ensemble/bounce_scaled/select)대로 계산한 값. 시뮬은 받은 값만 표시.
        const p = resp.churn_probability;
        const headline = p <= 1 ? p * 100 : p;
        const rounded = Math.round(headline * 10) / 10;
        // 새 값이 0(세션 이벤트 없음 등)이면 직전 퍼센트를 유지 → 0%로 깜빡이지 않게
        setChurnRate(prev => (rounded <= 0 && prev !== null ? prev : rounded));
        setHistory(h => [...h, rounded].slice(-MAX_POINTS));   // 선그래프 추이 누적
        setError(null);
      } catch (err) {
        if (!active) return;
        setChurnRate(null);
        setError(err instanceof BackendDisconnectedError
          ? '백엔드 미연결 — 실확률 불가'
          : '이탈확률 조회 실패');
      } finally {
        if (active) setIsLoading(false);
      }
    };

    // 즉시 1회 + 이후 4초마다 백엔드 재조회
    fetchChurnRate();
    const interval = setInterval(fetchChurnRate, refreshIntervalSec * 1000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [session.sessionId, session.userId, refreshIntervalSec]);

  const getChurnColor = (rate: number | null) => {
    if (rate === null) return 'text-slate-400';
    if (rate < 30) return 'text-green-400';
    if (rate < 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getChurnBgColor = (rate: number | null) => {
    if (rate === null) return 'bg-slate-700/20';
    if (rate < 30) return 'bg-green-500/10';
    if (rate < 60) return 'bg-yellow-500/10';
    return 'bg-red-500/10';
  };

  // 막대 색 — 정적 클래스로 반환(Tailwind가 동적 .replace() 클래스는 생성 못 해 막대가 안 차 보이던 버그 수정)
  const getBarColor = (rate: number | null) => {
    if (rate === null) return 'bg-slate-500';
    if (rate < 30) return 'bg-green-400';
    if (rate < 60) return 'bg-yellow-400';
    return 'bg-red-400';
  };

  const getMetricLabel = (key: string, fallback: string) => {
    if (key === 'churn_7d') return '7일 이탈(집계모델)';
    if (key === 'hazard') return '실시간 하자드';
    if (key === 'bounce') return '이탈 Bounce(30분)';
    return fallback;
  };

  return (
    <div className="fixed left-6 bottom-6 z-40 w-80">
      <Card className={`bg-slate-800 border-slate-700 transition-all duration-300 ${
        isCollapsed ? 'p-3' : 'p-4'
      }`}>
        {/* Header with Toggle */}
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex items-center gap-2 flex-1">
              <TrendingDown className="w-5 h-5 text-slate-400" />
              <h3 className="font-semibold text-slate-100">Churn Rate</h3>
            </div>
          )}
          
          {/* Churn Rate Display (Always Visible) */}
          <div className={`flex items-center gap-2 ${isCollapsed ? 'flex-1 justify-center' : ''}`}>
            {isLoading && (
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
            )}
            {churnRate !== null && !error && (
              <div className={`font-bold text-lg ${getChurnColor(churnRate)}`}>
                {churnRate.toFixed(1)}%
              </div>
            )}
          </div>

          {/* Toggle Button */}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-slate-400 hover:text-slate-100 transition-colors ml-2"
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Expanded Content */}
        {!isCollapsed && (
          <>
            <div className={`rounded-lg p-4 mt-3 ${getChurnBgColor(churnRate)}`}>
              {error ? (
                <div className="flex items-center gap-2 text-red-400 text-sm">
                  <AlertCircle className="w-4 h-4" />
                  <span>{error}</span>
                </div>
              ) : churnRate === null ? (
                <div className="text-slate-400 text-sm">Loading...</div>
              ) : (
                <div>
                  <p className="text-xs text-slate-400">
                    Estimated churn probability for this session
                  </p>
                </div>
              )}
            </div>

            {/* Churn Rate 추이 — 시계열 선그래프(바게이지 대체) */}
            {churnRate !== null && !error && (() => {
              const W = 288, H = 60;
              const pts = history.map((v, i) => {
                const x = history.length <= 1 ? W : (i / (history.length - 1)) * W;
                const y = H - (Math.min(Math.max(v, 0), 100) / 100) * H;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              }).join(' ');
              const stroke = churnRate < 30 ? '#4ade80' : churnRate < 60 ? '#facc15' : '#f87171';
              const lastX = history.length ? (history.length <= 1 ? W : W) : W;
              const lastY = H - (Math.min(Math.max(churnRate, 0), 100) / 100) * H;
              return (
                <div className="mt-4">
                  <div className="flex items-center justify-between text-[10px] text-slate-500 mb-0.5">
                    <span>Churn Rate 추이</span><span>최근 {history.length}p · {refreshIntervalSec}s 간격</span>
                  </div>
                  <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-16 bg-slate-900/40 rounded" preserveAspectRatio="none">
                    {/* 위험 임계 점선(60%·30%) */}
                    <line x1="0" y1={H * 0.4} x2={W} y2={H * 0.4} stroke="#475569" strokeWidth="0.5" strokeDasharray="3" vectorEffect="non-scaling-stroke" />
                    <line x1="0" y1={H * 0.7} x2={W} y2={H * 0.7} stroke="#475569" strokeWidth="0.5" strokeDasharray="3" vectorEffect="non-scaling-stroke" />
                    {history.length > 1 && (
                      <polyline points={pts} fill="none" stroke={stroke} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
                    )}
                    {history.length > 0 && <circle cx={lastX} cy={lastY} r="3" fill={stroke} vectorEffect="non-scaling-stroke" />}
                  </svg>
                  <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                    <span>0%</span><span className="text-slate-600">— 60% / 30% 위험선</span><span>100%</span>
                  </div>
                </div>
              );
            })()}

            {/* 3종 이탈값 (7일 churn · 하자드 · bounce) */}
            {breakdown.length > 0 && !error && (
              <div className="mt-4 space-y-2">
                <p className="text-xs text-slate-400 font-medium">이탈 지표 3종</p>
                {breakdown.map(b => {
                  const v = b.probability;
                  return (
                    <div key={b.key} className="flex items-center gap-2">
                      <span className="text-xs text-slate-300 w-32 shrink-0">{getMetricLabel(b.key, b.label)}</span>
                      <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${getBarColor(v)}`}
                          style={{ width: v === null ? '0%' : `${Math.min(v, 100)}%` }}
                        />
                      </div>
                      <span className={`text-xs font-semibold w-12 text-right ${getChurnColor(v)}`}>
                        {v === null ? 'N/A' : `${v.toFixed(1)}%`}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Info */}
            <p className="text-xs text-slate-500 mt-4">
              이탈예측 백엔드 실추론 · 세션 활동 기반 {refreshIntervalSec}초마다 갱신
            </p>
          </>
        )}
      </Card>
    </div>
  );
}
