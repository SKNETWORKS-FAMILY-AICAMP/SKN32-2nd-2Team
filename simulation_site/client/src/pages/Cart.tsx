import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useCart } from '@/contexts/CartContext';
import { logRemoveFromCartEvent, logPurchaseEvent } from '@/lib/eventLogger';
import { ArrowLeft, Trash2, Plus, Minus } from 'lucide-react';
import { toast } from 'sonner';

export default function Cart() {
  const [, setLocation] = useLocation();
  const { items, removeItem, updateQuantity, getTotalPrice, getTotalItems, clearCart } = useCart();

  const handleRemoveItem = async (productId: string) => {
    const item = items.find(i => i.productId === productId);
    if (item) {
      await logRemoveFromCartEvent(
        item.productId,
        item.categoryId,
        item.brand,
        item.price,
        item.quantity,
        '/cart'
      );
    }
    removeItem(productId);
    toast.success('Item removed from cart');
  };

  const handleCheckout = async () => {
    if (items.length === 0) {
      toast.error('Cart is empty');
      return;
    }

    // Log purchase event for all items
    await logPurchaseEvent(
      items.map(item => ({
        productId: item.productId,
        categoryId: item.categoryId,
        brand: item.brand,
        price: item.price,
        quantity: item.quantity,
      })),
      '/cart'
    );

    toast.success('Purchase completed!');
    clearCart();
    setLocation('/products');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-4">
          <Button
            variant="ghost"
            className="text-slate-400 hover:text-slate-100"
            onClick={() => setLocation('/products')}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Continue Shopping
          </Button>
        </div>
      </div>

      {/* Cart Content */}
      <div className="container mx-auto px-4 py-12">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 mb-8">
          Shopping Cart
        </h1>

        {items.length === 0 ? (
          <Card className="bg-slate-800 border-slate-700 p-12 text-center">
            <p className="text-slate-400 text-lg mb-6">Your cart is empty</p>
            <Button
              className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white"
              onClick={() => setLocation('/products')}
            >
              Start Shopping
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Cart Items */}
            <div className="lg:col-span-2 space-y-4">
              {items.map(item => (
                <Card
                  key={item.productId}
                  className="bg-slate-800 border-slate-700 p-6 flex items-center justify-between hover:border-purple-500 transition-colors"
                >
                  <div className="flex-1">
                    <h3 className="font-semibold text-slate-100 mb-1">{item.name}</h3>
                    <p className="text-sm text-slate-500 mb-3">
                      {item.brand.toUpperCase()} • {item.category}
                    </p>
                    <p className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                      ${item.price.toFixed(2)}
                    </p>
                  </div>

                  <div className="flex items-center gap-4 mx-6">
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-slate-700 text-slate-400 hover:text-slate-100"
                      onClick={() => updateQuantity(item.productId, Math.max(1, item.quantity - 1))}
                    >
                      <Minus className="w-3 h-3" />
                    </Button>
                    <Input
                      type="number"
                      min="1"
                      value={item.quantity}
                      onChange={(e) =>
                        updateQuantity(item.productId, Math.max(1, parseInt(e.target.value) || 1))
                      }
                      className="w-16 bg-slate-700 border-slate-600 text-center text-slate-100"
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-slate-700 text-slate-400 hover:text-slate-100"
                      onClick={() => updateQuantity(item.productId, item.quantity + 1)}
                    >
                      <Plus className="w-3 h-3" />
                    </Button>
                  </div>

                  <div className="text-right">
                    <p className="text-sm text-slate-400 mb-2">Subtotal</p>
                    <p className="text-lg font-bold text-slate-100 mb-4">
                      ${(item.price * item.quantity).toFixed(2)}
                    </p>
                    <Button
                      size="sm"
                      variant="destructive"
                      className="bg-red-900/50 hover:bg-red-900 text-red-400"
                      onClick={() => handleRemoveItem(item.productId)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>

            {/* Cart Summary */}
            <div>
              <Card className="bg-slate-800 border-slate-700 p-6 sticky top-24">
                <h2 className="text-xl font-bold text-slate-100 mb-6">Order Summary</h2>

                <div className="space-y-4 mb-6 pb-6 border-b border-slate-700">
                  <div className="flex justify-between text-slate-400">
                    <span>Items ({getTotalItems()})</span>
                    <span>${getTotalPrice().toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Shipping</span>
                    <span>Free</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Tax</span>
                    <span>Included</span>
                  </div>
                </div>

                <div className="flex justify-between items-center mb-6">
                  <span className="text-lg font-bold text-slate-100">Total</span>
                  <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                    ${getTotalPrice().toFixed(2)}
                  </span>
                </div>

                <Button
                  size="lg"
                  className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white mb-3"
                  onClick={handleCheckout}
                >
                  Proceed to Checkout
                </Button>

                <Button
                  size="lg"
                  variant="outline"
                  className="w-full border-slate-700 text-slate-400 hover:text-slate-100"
                  onClick={() => setLocation('/products')}
                >
                  Continue Shopping
                </Button>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
