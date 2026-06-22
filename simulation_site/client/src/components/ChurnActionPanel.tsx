import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { SessionManager, EventLogger, logViewEvent } from '@/lib/eventLogger';
import { getChurnPrediction, ChurnAction } from '@/lib/fastApiClient';
import { Sparkles, Tag, Share2 } from 'lucide-react';

/**
 * 이탈방지 액션 패널 — 백엔드 churn 예측의 recommended_action을 읽어 개입 UI를 렌더.
 * 시나리오: sns_view(첫진입·장기미접속) / discount_related(담고 미구매) / discount(조회만).
 * SNS 클릭 = view 이벤트 기록 → 다음 예측에서 이탈률 하락 기대.
 */
export default function ChurnActionPanel() {
  const [action, setAction] = useState<ChurnAction | null>(null);
  const session = SessionManager.getOrCreateSession();

  useEffect(() => {
    let active = true;
    const tick = async () => {
      try {
        const raw = await EventLogger.getSessionEvents(session.sessionId);
        const events = (raw as Record<string, unknown>[]).map(e => ({
          event_type: String(e.eventType ?? e.event_type ?? 'view'),
          product_id: String(e.productId ?? e.product_id ?? ''),
          category_id: String(e.categoryId ?? e.category_id ?? ''),
          brand: String(e.brand ?? ''),
          price: Number(e.price ?? 0),
          quantity: Number(e.quantity ?? 1),
          timestamp: String(e.eventTime ?? e.event_time ?? new Date().toISOString()),
        }));
        const churnSid = `${session.sessionId}:act${Date.now()}`;
        const resp = await getChurnPrediction(churnSid, session.userId, events);
        if (!active) return;
        const a = resp.recommended_action ?? null;
        setAction(a && a.action_type !== 'none' ? a : null);
      } catch {
        if (active) setAction(null); // 백엔드 미연결이면 패널 숨김(가짜값 금지)
      }
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => { active = false; clearInterval(id); };
  }, [session.sessionId, session.userId]);

  if (!action) return null;

  const onSnsClick = () => {
    // SNS 둘러보기 = view 이벤트로 취급 → 인게이지먼트↑ → 이탈률↓
    logViewEvent('sns_promo', 'sns', 'GAJIMA', 0, '/sns', 'churn_action:sns_view');
    setAction(null);
  };

  // 쿠폰 받기/할인상품 보기 = 해당(연관)상품 view 이벤트로 기록 → 인게이지먼트↑ → 다음 예측에서 이탈률↓
  const onCouponClick = () => {
    const rel = action.payload?.related?.[0];
    const cat = String(rel?.category_id ?? 'promo');
    logViewEvent('coupon_item', cat, 'GAJIMA', 0, '/coupon', `churn_action:${action.action_type}`);
    setAction(null);
  };

  const discount = action.payload?.discount_pct;

  return (
    <div className="fixed right-6 bottom-6 z-40 w-80">
      <Card className="bg-slate-800 border-amber-500/40 p-4 shadow-xl">
        <div className="flex items-center gap-2 mb-2">
          {action.action_type === 'sns_view' && <Share2 className="w-5 h-5 text-cyan-400" />}
          {action.action_type === 'discount_related' && <Tag className="w-5 h-5 text-pink-400" />}
          {action.action_type === 'discount' && <Sparkles className="w-5 h-5 text-amber-400" />}
          <h3 className="font-semibold text-slate-100 text-sm">이탈 방지 제안</h3>
        </div>
        <p className="text-sm text-slate-300 mb-3">{action.message}</p>

        {action.action_type === 'sns_view' && (
          <button
            onClick={onSnsClick}
            className="w-full py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-sm font-medium hover:opacity-90"
          >
            SNS 인기상품 둘러보기 →
          </button>
        )}

        {(action.action_type === 'discount_related' || action.action_type === 'discount') && discount != null && (
          <button
            onClick={onCouponClick}
            className="w-full rounded-lg bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-amber-300 text-sm font-semibold text-center hover:bg-amber-500/20 transition-colors"
          >
            🎟️ {discount}% 할인 쿠폰 받고 상품 보기 →
          </button>
        )}

        {action.action_type === 'discount_related' && action.payload?.related?.length ? (
          <div className="mt-3">
            <p className="text-xs text-slate-400 mb-1">연관 카테고리 추천</p>
            <div className="flex flex-wrap gap-1">
              {action.payload.related.slice(0, 3).map((r, i) => (
                <span key={i} className="text-xs bg-slate-700 text-slate-200 px-2 py-1 rounded">
                  cat {String(r.category_id)}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        <p className="text-[10px] text-slate-500 mt-3">trigger: {action.trigger}</p>
      </Card>
    </div>
  );
}
