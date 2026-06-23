import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { getRecommendations, RecommendedProduct } from '@/lib/fastApiClient';
import { SessionManager } from '@/lib/eventLogger';
import { Sparkles } from 'lucide-react';

interface RecommendedProductsProps {
  productId: string;
  categoryId: string;
  brand: string;
  onProductSelect: (productId: string) => void;
}

export default function RecommendedProducts({
  productId,
  categoryId,
  brand,
  onProductSelect,
}: RecommendedProductsProps) {
  const [recommendations, setRecommendations] = useState<RecommendedProduct[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const session = SessionManager.getOrCreateSession();

  useEffect(() => {
    const fetchRecommendations = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await getRecommendations(
          session.sessionId,
          session.userId,
          productId,
          categoryId,
          brand
        );
        setRecommendations(response.recommendations || []);
      } catch (err) {
        setError('Failed to load recommendations');
        console.error('Recommendations error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecommendations();
  }, [productId, categoryId, brand, session.sessionId, session.userId]);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-slate-100">Recommended For You</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(i => (
            <Skeleton key={i} className="h-32 bg-slate-700" />
          ))}
        </div>
      </div>
    );
  }

  if (error || recommendations.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-purple-400" />
        <h3 className="text-lg font-semibold text-slate-100">Recommended For You</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {recommendations.map(product => (
          <Card
            key={product.product_id}
            className="bg-slate-800 border-slate-700 hover:border-cyan-500 transition-all duration-300 overflow-hidden group cursor-pointer"
            onClick={() => onProductSelect(product.product_id)}
          >
            <div className="p-4">
              {/* Product Image Placeholder */}
              <div className="w-full h-24 bg-gradient-to-br from-slate-700 to-slate-900 flex items-center justify-center rounded mb-3 group-hover:from-cyan-600/20 group-hover:to-purple-600/20 transition-all duration-300">
                <span className="text-3xl">💄</span>
              </div>

              {/* Product Info */}
              <h4 className="font-semibold text-slate-100 mb-1 line-clamp-2 group-hover:text-cyan-400 transition-colors">
                {product.name}
              </h4>
              <p className="text-xs text-slate-500 mb-1">{product.brand.toUpperCase()}</p>

              {/* Score Badge */}
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-slate-400">Match: {(product.score * 100).toFixed(0)}%</span>
                <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded">
                  {product.reason}
                </span>
              </div>

              {/* Price and Button */}
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                  ${product.price.toFixed(2)}
                </span>
                <Button
                  size="sm"
                  className="bg-cyan-500 hover:bg-cyan-600 text-white text-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    onProductSelect(product.product_id);
                  }}
                >
                  View
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
