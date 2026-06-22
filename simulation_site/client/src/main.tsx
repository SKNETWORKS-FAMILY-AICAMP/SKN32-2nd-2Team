import { trpc } from "@/lib/trpc";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { httpBatchLink } from "@trpc/client";
import { createRoot } from "react-dom/client";
import superjson from "superjson";
import App from "./App";
import "./index.css";

// 애널리틱스(umami) — env가 '실제 http(s) URL'일 때만 동적 주입(미설정/플레이스홀더면 스킵 → URL 깨짐·서버다운 방지).
const _analyticsUrl = import.meta.env.VITE_ANALYTICS_ENDPOINT;
if (typeof _analyticsUrl === "string" && /^https?:\/\//.test(_analyticsUrl)) {
  const s = document.createElement("script");
  s.defer = true;
  s.src = `${_analyticsUrl.replace(/\/$/, "")}/umami`;
  const wid = import.meta.env.VITE_ANALYTICS_WEBSITE_ID;
  if (wid) s.setAttribute("data-website-id", String(wid));
  document.head.appendChild(s);
}

const queryClient = new QueryClient();

// 인증/리다이렉트 로직 제거(마누스 OAuth 없음). 에러는 콘솔 로깅만.
queryClient.getQueryCache().subscribe(event => {
  if (event.type === "updated" && event.action.type === "error") {
    console.error("[API Query Error]", event.query.state.error);
  }
});

queryClient.getMutationCache().subscribe(event => {
  if (event.type === "updated" && event.action.type === "error") {
    console.error("[API Mutation Error]", event.mutation.state.error);
  }
});

const trpcClient = trpc.createClient({
  links: [
    httpBatchLink({
      url: "/api/trpc",
      transformer: superjson,
      fetch(input, init) {
        return globalThis.fetch(input, {
          ...(init ?? {}),
          credentials: "include",
        });
      },
    }),
  ],
});

createRoot(document.getElementById("root")!).render(
  <trpc.Provider client={trpcClient} queryClient={queryClient}>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </trpc.Provider>
);
