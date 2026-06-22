import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { EventLogger, SessionManager } from '@/lib/eventLogger';
import { ChevronDown, ChevronUp, X } from 'lucide-react';

interface Event {
  id?: number;
  eventId: string;
  userId: string;
  sessionId: string;
  eventType: string;
  eventTime: string;
  productId: string;
  categoryId: string;
  brand: string;
  price: number;
  quantity: number;
  pageUrl: string;
  referrer?: string;
  deviceType: string;
  payloadJson?: Record<string, unknown>;
  createdAt?: string;
}

interface EventLogViewerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function EventLogViewer({ isOpen, onClose }: EventLogViewerProps) {
  const [events, setEvents] = useState<Event[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const session = SessionManager.getOrCreateSession();

  // Fetch events on mount and when session changes
  useEffect(() => {
    if (isOpen) {
      fetchEvents();
      // Refresh every 2 seconds
      const interval = setInterval(fetchEvents, 2000);
      return () => clearInterval(interval);
    }
  }, [isOpen, session.sessionId]);

  const fetchEvents = async () => {
    setIsLoading(true);
    try {
      const data = await EventLogger.getSessionEvents(session.sessionId);
      // Sort by event time (newest first)
      const sorted = (data as Event[]).sort((a, b) => {
        const timeA = new Date(a.eventTime || a.createdAt || 0).getTime();
        const timeB = new Date(b.eventTime || b.createdAt || 0).getTime();
        return timeB - timeA;
      });
      setEvents(sorted);
    } catch (error) {
      console.error('Failed to fetch events:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  const getEventColor = (eventType: string) => {
    switch (eventType) {
      case 'view':
        return 'text-cyan-400';
      case 'cart':
        return 'text-green-400';
      case 'remove_from_cart':
        return 'text-yellow-400';
      case 'purchase':
        return 'text-purple-400';
      default:
        return 'text-slate-400';
    }
  };

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'view':
        return '👁️';
      case 'cart':
        return '🛒';
      case 'remove_from_cart':
        return '❌';
      case 'purchase':
        return '✅';
      default:
        return '📝';
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/50"
        onClick={onClose}
      />

      {/* Panel */}
      <Card className="fixed right-0 top-0 bottom-0 z-50 w-96 bg-slate-800 border-l border-slate-700 rounded-none shadow-2xl flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-700 p-4 flex items-center justify-between">
          <h2 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
            Event Log
          </h2>
          <Button
            size="sm"
            variant="ghost"
            className="text-slate-400 hover:text-slate-100"
            onClick={onClose}
          >
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Session Info */}
        <div className="border-b border-slate-700 p-4 bg-slate-700/30">
          <p className="text-xs text-slate-400 mb-1">Session ID</p>
          <p className="text-sm font-mono text-slate-200 truncate">{session.sessionId}</p>
          <p className="text-xs text-slate-400 mt-2">Total Events: {events.length}</p>
        </div>

        {/* Events List — min-h-0 로 flex 자식이 줄어들어야 ScrollArea 내부 스크롤이 동작(아래 항목 잘림 방지) */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-4 space-y-2">
            {events.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-slate-500 text-sm">No events yet</p>
              </div>
            ) : (
              events.map((event) => (
                <div
                  key={event.eventId}
                  className="bg-slate-700/50 border border-slate-600 rounded-lg overflow-hidden"
                >
                  {/* Event Summary */}
                  <button
                    onClick={() =>
                      setExpandedId(expandedId === event.eventId ? null : event.eventId)
                    }
                    className="w-full p-3 text-left hover:bg-slate-700/70 transition-colors flex items-center justify-between"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <span className="text-lg">{getEventIcon(event.eventType)}</span>
                      <div className="flex-1 min-w-0">
                        <p className={`font-semibold text-sm ${getEventColor(event.eventType)}`}>
                          {event.eventType.toUpperCase()}
                        </p>
                        <p className="text-xs text-slate-400 truncate">
                          {event.productId} • ₩{event.price.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    {expandedId === event.eventId ? (
                      <ChevronUp className="w-4 h-4 text-slate-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-slate-400" />
                    )}
                  </button>

                  {/* Event Details */}
                  {expandedId === event.eventId && (
                    <div className="border-t border-slate-600 p-3 bg-slate-800/50 text-xs space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-slate-500">Event ID</p>
                          <p className="text-slate-300 font-mono truncate">{event.eventId}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">User ID</p>
                          <p className="text-slate-300 font-mono truncate">{event.userId}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">Time</p>
                          <p className="text-slate-300">
                            {new Date(event.eventTime || event.createdAt || 0).toLocaleTimeString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-slate-500">Device</p>
                          <p className="text-slate-300">{event.deviceType}</p>
                        </div>
                      </div>

                      <div>
                        <p className="text-slate-500">Product</p>
                        <p className="text-slate-300">{event.productId}</p>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-slate-500">Category</p>
                          <p className="text-slate-300">{event.categoryId}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">Brand</p>
                          <p className="text-slate-300">{event.brand}</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-slate-500">Price</p>
                          <p className="text-slate-300">₩{event.price.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">Quantity</p>
                          <p className="text-slate-300">{event.quantity}</p>
                        </div>
                      </div>

                      <div>
                        <p className="text-slate-500">Page URL</p>
                        <p className="text-slate-300 truncate">{event.pageUrl}</p>
                      </div>

                      {event.referrer && (
                        <div>
                          <p className="text-slate-500">Referrer</p>
                          <p className="text-slate-300 truncate">{event.referrer}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="border-t border-slate-700 p-4 bg-slate-700/30">
          <Button
            size="sm"
            className="w-full bg-cyan-500 hover:bg-cyan-600 text-white"
            onClick={fetchEvents}
            disabled={isLoading}
          >
            {isLoading ? 'Refreshing...' : 'Refresh Events'}
          </Button>
        </div>
      </Card>
    </>
  );
}
