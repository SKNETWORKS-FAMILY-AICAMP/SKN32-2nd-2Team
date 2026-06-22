/**
 * FastAPI 클라이언트 유틸리티
 * 외부 FastAPI 서버와 통신하기 위한 함수들
 */

// 백엔드 URL 해석: 설정(공개/오라클) → 안 닿으면 FALLBACK(로컬) → 둘 다 실패면 각 호출이 mock 폴백.
// "공인 IP 판별"이 아니라 실제 도달 가능성(/health)으로 결정한다.
const PRIMARY_URL = import.meta.env.VITE_FASTAPI_URL || 'http://localhost:8090';
const FALLBACK_URL = import.meta.env.VITE_FASTAPI_FALLBACK_URL || '';

let _resolvedBase: string | null = null;

async function _reachable(url: string): Promise<boolean> {
  try {
    const r = await fetch(`${url}/health`, { method: 'GET' });
    return r.ok;
  } catch {
    return false;
  }
}

/** PRIMARY → FALLBACK 순으로 헬스체크해 도달 가능한 베이스 URL을 반환. 둘 다 안 닿으면 null.
 *  (mock으로 위장하지 않음 — "양쪽 다 연결돼야 진행" 원칙) */
async function resolveBaseUrl(): Promise<string | null> {
  if (_resolvedBase) return _resolvedBase;
  for (const url of [PRIMARY_URL, FALLBACK_URL].filter(Boolean)) {
    if (await _reachable(url)) {
      _resolvedBase = url;
      return url;
    }
  }
  return null;   // 미연결 — 캐시 안 함(다음 호출에서 재시도)
}

// ── 이벤트 흐름 모드 (19-2/19-3/05-6-4 정합) ────────────────────────────────
// 기본 = B(neon): 이벤트를 백엔드로 보내 Neon sim_event_log에 영속(단일원천).
// Neon 미연결 = A-최소(direct): 클라 버퍼 + /api/churn/predict 무상태 채점(폴백).
// VITE_SIM_MODE = auto | direct | neon (기본 auto: /health.neon 으로 판별).
export type SimMode = 'direct' | 'neon';
const SIM_MODE_ENV = (import.meta.env.VITE_SIM_MODE || 'auto') as 'auto' | SimMode;
let _simMode: SimMode | null = null;

/** 백엔드 /health.neon 으로 모드 결정(캐시). auto가 아니면 env 값 사용. */
export async function resolveSimMode(): Promise<SimMode> {
  if (_simMode) return _simMode;
  if (SIM_MODE_ENV !== 'auto') { _simMode = SIM_MODE_ENV; return _simMode; }
  try {
    const base = await resolveBaseUrl();
    if (base) {
      const r = await fetch(`${base}/health`, { method: 'GET' });
      if (r.ok) {
        const j = await r.json();
        const neon = j?.data?.neon === true;     // 봉투: { data: { neon } }
        _simMode = neon ? 'neon' : 'direct';
        return _simMode;
      }
    }
  } catch { /* 무시 → direct 폴백 */ }
  return 'direct';   // 미연결/판별 실패 → 캐시 안 함, 다음에 재시도
}

/** 동기 조회(미해석이면 direct). resolveSimMode()가 먼저 한 번 호출돼 있어야 정확. */
export function currentSimMode(): SimMode {
  return _simMode || 'direct';
}

/** 라이브 헬스체크(캐시 안 함) — 배너/상태표시용. PRIMARY→FALLBACK 중 하나라도 닿으면 true. */
export async function pingBackend(): Promise<boolean> {
  for (const url of [PRIMARY_URL, FALLBACK_URL].filter(Boolean)) {
    if (await _reachable(url)) return true;
  }
  return false;
}

/** 백엔드 연결 여부(프론트는 이미 로드됨 → 백엔드 도달 가능성만 확인). UI 게이트용. */
export async function isBackendConnected(): Promise<boolean> {
  return pingBackend();
}

/** 미연결 시 던지는 에러(각 호출이 가짜 데이터 대신 명확히 실패). */
export class BackendDisconnectedError extends Error {
  constructor() {
    super('BACKEND_DISCONNECTED');
    this.name = 'BackendDisconnectedError';
  }
}

export interface ChurnPredictionRequest {
  session_id: string;
  user_id: string;
  events: Array<{
    event_type: string;
    product_id: string;
    category_id: string;
    brand: string;
    price: number;
    quantity: number;
    timestamp: string;
  }>;
}

export interface ChurnAction {
  action_type: 'sns_view' | 'discount_related' | 'discount' | 'none';
  trigger: string;
  message: string;
  payload: {
    sns_url?: string;
    as_view_event?: boolean;
    discount_pct?: number;
    related?: Array<{ category_id?: string | number; cosine?: number }>;
  };
}

export interface ChurnPredictionResponse {
  session_id: string;
  churn_probability: number;
  risk_level: 'low' | 'medium' | 'high';
  recommended_action?: ChurnAction;
  timestamp: string;
}

export interface RecommendedProduct {
  product_id: string;
  name: string;
  category_id: string;
  brand: string;
  price: number;
  score: number;
  reason: string;
}

export interface RecommendationResponse {
  session_id: string;
  user_id: string;
  current_product_id: string;
  recommendations: RecommendedProduct[];
  timestamp: string;
}

/**
 * 현재 세션의 이탈 확률 조회
 */
export async function getChurnPrediction(
  sessionId: string,
  userId: string,
  events: ChurnPredictionRequest['events'] = []
): Promise<ChurnPredictionResponse> {
  const base = await resolveBaseUrl();
  if (!base) throw new BackendDisconnectedError();   // 미연결 → 가짜(랜덤) 대신 명확히 실패
  const response = await fetch(`${base}/api/churn/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, user_id: userId, events } as ChurnPredictionRequest),
  });
  if (!response.ok) throw new Error(`churn predict failed: ${response.status}`);
  return await response.json();
}

/**
 * 추천 상품 조회
 */
export async function getRecommendations(
  sessionId: string,
  userId: string,
  currentProductId: string,
  categoryId: string,
  brand: string
): Promise<RecommendationResponse> {
  const base = await resolveBaseUrl();
  if (!base) throw new BackendDisconnectedError();   // 미연결 → 빈/가짜 대신 명확히 실패(호출부가 에러 표시)
  const response = await fetch(`${base}/api/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId, user_id: userId,
      current_product_id: currentProductId, category_id: categoryId, brand,
    }),
  });
  if (!response.ok) throw new Error(`recommendations failed: ${response.status}`);
  return await response.json();
}

/**
 * 이벤트 로그 전송 (FastAPI 서버에 저장)
 */
export async function sendEventToFastAPI(eventData: {
  event_id: string;
  user_id: string;
  session_id: string;
  event_type: string;
  event_time: string;
  product_id: string;
  category_id: string;
  brand: string;
  price: number;
  quantity: number;
  page_url: string;
  referrer?: string;
  device_type: string;
  payload_json?: Record<string, unknown>;
}): Promise<void> {
  const base = await resolveBaseUrl();
  if (!base) return;          // 미연결 → 이벤트 전송 스킵(best-effort, 가짜 생성 안 함)
  try {
    await fetch(`${base}/api/events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(eventData),
    });
  } catch (error) {
    console.error('Event send error:', error);   // best-effort: 실패해도 앱은 계속
  }
}

/**
 * FastAPI 서버 상태 확인
 */
export async function checkFastAPIHealth(): Promise<boolean> {
  return isBackendConnected();   // PRIMARY→FALLBACK 도달 가능 여부
}

/**
 * 세션 분석 데이터 조회
 */
export async function getSessionAnalytics(sessionId: string): Promise<{
  session_id: string;
  total_events: number;
  event_breakdown: Record<string, number>;
  average_session_duration: number;
  products_viewed: number;
  products_in_cart: number;
  purchases: number;
}> {
  const base = await resolveBaseUrl();
  if (!base) throw new BackendDisconnectedError();
  const response = await fetch(`${base}/api/analytics/session/${sessionId}`, { method: 'GET' });
  if (!response.ok) throw new Error(`session analytics failed: ${response.status}`);
  return await response.json();
}

export interface CatalogProduct {
  productId: string;
  sku: string;
  name: string;
  categoryId: string;
  category: string;
  brand: string;
  price: number;
}

/** REES46 seed 카탈로그 상품(인기순+필터). 백엔드 /api/catalog/products. 정적 productData 대체. */
export async function getCatalogProducts(
  opts: { limit?: number; category?: string; brand?: string; q?: string } = {}
): Promise<CatalogProduct[]> {
  const base = await resolveBaseUrl();
  if (!base) throw new BackendDisconnectedError();
  const p = new URLSearchParams();
  p.set('limit', String(opts.limit ?? 60));
  if (opts.category && opts.category !== 'all') p.set('category', opts.category);
  if (opts.brand && opts.brand !== 'all') p.set('brand', opts.brand);
  if (opts.q) p.set('q', opts.q);
  const r = await fetch(`${base}/api/catalog/products?${p.toString()}`);
  if (!r.ok) throw new Error(`catalog products failed: ${r.status}`);
  const data = await r.json();
  return (data.products ?? []).map((x: Record<string, unknown>) => ({
    productId: String(x.product_id),
    sku: String(x.product_id),
    name: String(x.name),
    categoryId: String(x.category_id),
    category: String(x.category_name),
    brand: String(x.brand),
    price: Number(x.price),
  }));
}

/** 드롭다운용 카테고리/브랜드. 백엔드 /api/catalog/facets. */
export async function getCatalogFacets(): Promise<{ categories: { category_id: string; name: string }[]; brands: string[] }> {
  const base = await resolveBaseUrl();
  if (!base) throw new BackendDisconnectedError();
  const r = await fetch(`${base}/api/catalog/facets`);
  if (!r.ok) throw new Error(`catalog facets failed: ${r.status}`);
  return await r.json();
}
