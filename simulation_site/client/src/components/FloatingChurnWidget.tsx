import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { SessionManager, EventLogger } from '@/lib/eventLogger';
import { getChurnPrediction, getActiveUserState, BackendDisconnectedError, ChurnBreakdownItem } from '@/lib/fastApiClient';
import { AlertCircle, TrendingDown, ChevronRight, ChevronLeft } from 'lucide-react';

export default function FloatingChurnWidget() {
  const [churnRate, setChurnRate] = useState<number | null>(null);
  const [breakdown, setBreakdown] = useState<ChurnBreakdownItem[]>([]);
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

            {/* Progress Bar */}
            {churnRate !== null && !error && (
              <div className="mt-4">
                <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      churnRate < 30
                        ? 'bg-gradient-to-r from-green-500 to-green-400'
                        : churnRate < 60
                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-400'
                        : 'bg-gradient-to-r from-red-500 to-red-400'
                    }`}
                    style={{ width: `${churnRate}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Low Risk</span>
                  <span>High Risk</span>
                </div>
              </div>
            )}

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
