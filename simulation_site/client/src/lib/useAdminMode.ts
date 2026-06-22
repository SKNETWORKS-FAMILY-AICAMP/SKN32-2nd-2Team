import { useSyncExternalStore } from 'react';

// 시뮬 어드민(이탈 메트릭 패널) 토글 — localStorage 영속 + 전역 구독.
// 기본 ON. App 레벨에서 마운트되므로 페이지 이동(상세·장바구니)에도 꺼지지 않는다.
const KEY = 'sim_admin_mode';
const listeners = new Set<() => void>();

function getSnapshot(): boolean {
  try {
    const v = localStorage.getItem(KEY);
    return v === null ? true : v === '1';   // 미설정 = 기본 켜짐
  } catch {
    return true;
  }
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  // 다른 탭/창에서 바뀐 것도 반영
  const onStorage = (e: StorageEvent) => { if (e.key === KEY) cb(); };
  window.addEventListener('storage', onStorage);
  return () => { listeners.delete(cb); window.removeEventListener('storage', onStorage); };
}

export function setAdminMode(on: boolean): void {
  try { localStorage.setItem(KEY, on ? '1' : '0'); } catch { /* 무시 */ }
  listeners.forEach(l => l());
}

export function useAdminMode(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => true);
}
