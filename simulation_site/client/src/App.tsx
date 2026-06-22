import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import { CartProvider } from "./contexts/CartContext";
import ChurnActionPanel from "./components/ChurnActionPanel";
import BackendStatusBanner from "./components/BackendStatusBanner";
import SimulationControl from "./components/SimulationControl";
import FloatingChurnWidget from "./components/FloatingChurnWidget";
import { useAdminMode } from "./lib/useAdminMode";
import Home from "./pages/Home";
import ProductList from "./pages/ProductList";
import ProductDetail from "./pages/ProductDetail";
import Cart from "./pages/Cart";

function Router() {
  // make sure to consider if you need authentication for certain routes
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/products" component={ProductList} />
      <Route path="/product/:id" component={ProductDetail} />
      <Route path="/cart" component={Cart} />
      <Route path="/404" component={NotFound} />
      {/* Final fallback route */}
      <Route component={NotFound} />
    </Switch>
  );
}

// NOTE: About Theme
// - First choose a default theme according to your design style (dark or light bg), than change color palette in index.css
//   to keep consistent foreground/background color across components
// - If you want to make theme switchable, pass `switchable` ThemeProvider and use `useTheme` hook

// 어드민 도구(시뮬 컨트롤 + 이탈 메트릭) — Router 밖(App 레벨) 마운트라 페이지 이동에도 언마운트되지 않음.
// 이탈 메트릭 패널은 어드민 토글(기본 ON, localStorage 영속)로 켜짐.
function GlobalAdminTools() {
  const admin = useAdminMode();
  return (
    <>
      <SimulationControl />
      {admin && <FloatingChurnWidget />}
    </>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider
        defaultTheme="dark"
        // switchable
      >
        <CartProvider>
          <TooltipProvider>
            <BackendStatusBanner />
            <Toaster />
            <Router />
            <ChurnActionPanel />
            <GlobalAdminTools />
          </TooltipProvider>
        </CartProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
