import { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { SAMPLE_PRODUCTS, CATEGORIES, BRANDS, getProductsByCategory, searchProducts } from '@/lib/productData';
import { getCatalogProducts, getCatalogFacets, getActiveUserState } from '@/lib/fastApiClient';
import { logViewEvent } from '@/lib/eventLogger';
import { SessionManager } from '@/lib/eventLogger';
import { ShoppingCart, Search, Eye } from 'lucide-react';
import EventLogViewer from '@/components/EventLogViewer';
import { useAdminMode, setAdminMode } from '@/lib/useAdminMode';

export default function ProductList() {
  const [, setLocation] = useLocation();
  const [filteredProducts, setFilteredProducts] = useState(SAMPLE_PRODUCTS);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedBrand, setSelectedBrand] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showEventLog, setShowEventLog] = useState(false);
  // 드롭다운 facet(실 카테고리/브랜드). 백엔드 미연결 시 정적 폴백.
  const [facetCats, setFacetCats] = useState<{ id: string; name: string }[]>(CATEGORIES);
  const [facetBrands, setFacetBrands] = useState<string[]>(BRANDS);
  // 접속 유저 ID: 대시보드가 설정한 active-user(서버) 우선, 없으면 로컬 세션 userId
  const [connectedUser, setConnectedUser] = useState<string>(() => SessionManager.getOrCreateSession().userId);
  const adminMode = useAdminMode();   // 전역·영속(기본 ON) — 페이지 이동에도 유지

  // 대시보드↔시뮬 동기화: active-user를 폴링해 헤더에 표시(대시보드에서 ID 설정 시 반영)
  useEffect(() => {
    let active = true;
    const sync = async () => {
      const state = await getActiveUserState();
      if (active && state.user_id) setConnectedUser(state.user_id);
    };
    sync();
    const t = setInterval(sync, 5000);
    return () => { active = false; clearInterval(t); };
  }, []);

  // Initialize session on mount
  useEffect(() => {
    SessionManager.getOrCreateSession();
    getCatalogFacets()
      .then(f => {
        if (f.categories?.length) setFacetCats(f.categories.map(c => ({ id: c.category_id, name: c.name })));
        if (f.brands?.length) setFacetBrands(f.brands);
      })
      .catch(() => { /* 미연결 → 정적 CATEGORIES/BRANDS 유지 */ });
  }, []);

  // 상품목록: 백엔드 REES46 카탈로그(seed 54k, 인기순+필터) 우선, 미연결 시 정적 폴백
  useEffect(() => {
    let active = true;
    const applyStatic = () => {
      let results = SAMPLE_PRODUCTS;
      if (selectedCategory && selectedCategory !== 'all') results = getProductsByCategory(selectedCategory);
      if (selectedBrand && selectedBrand !== 'all') results = results.filter(p => p.brand === selectedBrand);
      if (searchQuery) results = searchProducts(searchQuery);
      setFilteredProducts(results);
    };
    (async () => {
      try {
        const items = await getCatalogProducts({
          limit: 60, category: selectedCategory, brand: selectedBrand, q: searchQuery,
        });
        if (!active) return;
        if (items.length) setFilteredProducts(items as typeof SAMPLE_PRODUCTS);
        else applyStatic();
      } catch {
        if (active) applyStatic();   // 백엔드 미연결 → 정적 25개 폴백
      }
    })();
    return () => { active = false; };
  }, [selectedCategory, selectedBrand, searchQuery]);

  const handleProductClick = async (productId: string) => {
    // Log view event (동적 카탈로그/정적 폴백 모두 filteredProducts에서 조회)
    const product = filteredProducts.find(p => p.productId === productId) || SAMPLE_PRODUCTS.find(p => p.productId === productId);
    if (product) {
      await logViewEvent(
        product.productId,
        product.categoryId,
        product.brand,
        product.price,
        '/products',
        '/'
      ).catch(err => console.error('Failed to log view event:', err));
    }

    // Navigate to product detail
    setLocation(`/product/${productId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* 시뮬 컨트롤·이탈 메트릭은 App 레벨(GlobalAdminTools)에서 전역 마운트 — 페이지 이동에도 유지 */}

      {/* Event Log Viewer */}
      <EventLogViewer isOpen={showEventLog} onClose={() => setShowEventLog(false)} />

      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
            <div className="flex items-center gap-4 min-w-0">
              <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400">
                Cosmetics Shop
              </h1>
              <Button
                variant="ghost"
                className="text-slate-400 hover:text-slate-100"
                onClick={() => setLocation('/')}
              >
                Home
              </Button>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2 min-w-0">
              <Button
                variant="outline"
                className="border-cyan-500 text-cyan-400 hover:bg-cyan-500/10"
                onClick={() => setShowEventLog(!showEventLog)}
              >
                <Eye className="w-4 h-4 mr-2" />
                Events
              </Button>
              <Button
                variant="outline"
                className="border-purple-500 text-purple-400 hover:bg-purple-500/10"
                onClick={() => setLocation('/cart')}
              >
                <ShoppingCart className="w-4 h-4 mr-2" />
                Cart
              </Button>
              {/* 접속 유저 ID — 대시보드(active-user)와 동기화 표시 */}
              <div
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-emerald-500/60 bg-emerald-500/10 text-emerald-300 text-sm font-mono max-w-[240px]"
                title="대시보드에서 설정한 진단 대상 유저(서버 동기화)"
              >
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                <span className="break-all leading-snug min-w-0">User: {connectedUser}</span>
              </div>
              <div
                className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-slate-600 bg-slate-800/70 text-slate-200 text-sm shrink-0"
                title="Admin metric panel"
              >
                <span className="font-medium">Admin</span>
                <Switch
                  checked={adminMode}
                  onCheckedChange={setAdminMode}
                  aria-label="Toggle admin churn metrics"
                  className="data-[state=checked]:bg-cyan-500"
                />
              </div>
            </div>
          </div>

          {/* Search Bar */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-3 w-4 h-4 text-slate-500" />
            <Input
              placeholder="Search products..."
              className="pl-10 bg-slate-800 border-slate-700 text-slate-100 placeholder:text-slate-500"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-100">
                <SelectValue placeholder="Select Category" />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-700">
                <SelectItem value="all">All Categories</SelectItem>
                {facetCats.map(cat => (
                  <SelectItem key={cat.id} value={cat.id}>
                    {cat.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={selectedBrand} onValueChange={setSelectedBrand}>
              <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-100">
                <SelectValue placeholder="Select Brand" />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-700">
                <SelectItem value="all">All Brands</SelectItem>
                {facetBrands.map(brand => (
                  <SelectItem key={brand} value={brand}>
                    {brand.replace('_', ' ').toUpperCase()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Products Grid */}
      <div className="container mx-auto px-4 py-12">
        {filteredProducts.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-slate-400 text-lg">No products found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {filteredProducts.map(product => (
              <Card
                key={product.productId}
                className="group bg-slate-800 border-slate-700 hover:border-purple-500 cursor-pointer transition-all duration-300 overflow-hidden hover:shadow-lg hover:shadow-purple-500/20"
                onClick={() => handleProductClick(product.productId)}
              >
                {/* Product Image Placeholder */}
                <div className="w-full h-48 bg-gradient-to-br from-slate-700 to-slate-900 flex items-center justify-center group-hover:from-purple-600/20 group-hover:to-pink-600/20 transition-all duration-300">
                  <div className="text-center">
                    <div className="text-4xl mb-2">💄</div>
                    <p className="text-xs text-slate-500">{product.sku}</p>
                  </div>
                </div>

                {/* Product Info */}
                <div className="p-4">
                  <h3 className="font-semibold text-slate-100 mb-1 line-clamp-2 group-hover:text-purple-400 transition-colors">
                    {product.name}
                  </h3>
                  <p className="text-xs text-slate-500 mb-2">{product.brand.toUpperCase()}</p>
                  <p className="text-xs text-slate-400 mb-3">{product.category}</p>

                  {/* Price */}
                  <div className="flex items-center justify-between">
                    <span className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                      ₩{product.price.toLocaleString()}
                    </span>
                    <Button
                      size="sm"
                      className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleProductClick(product.productId);
                      }}
                    >
                      View
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
