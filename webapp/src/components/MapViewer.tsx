'use client';

import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { MapItem } from '@/types/api';
import { ApiService } from '@/lib/api';

interface MapViewerProps {
  map: MapItem;
  isOpen: boolean;
  onClose: () => void;
}

export function MapViewer({ map, isOpen, onClose }: MapViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showInNewTab, setShowInNewTab] = useState(false);

  const mapUrl = ApiService.getMapUrl(map.url);

  const handleIframeLoad = () => {
    setIsLoading(false);
    setError(null);
  };

  const handleIframeError = () => {
    setIsLoading(false);
    setError('Failed to load map. The map might not be accessible or there could be a network issue.');
  };

  const handleOpenInNewTab = () => {
    window.open(mapUrl, '_blank');
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl w-full h-[90vh] flex flex-col">
        <DialogHeader className="flex-shrink-0">
          <DialogTitle className="flex items-center gap-2">
            {map.name}
            <Badge variant="secondary">HTML Карта</Badge>
          </DialogTitle>
        </DialogHeader>
        
        <div className="flex-1 flex flex-col space-y-4 min-h-0">
          <div className="p-4 bg-muted rounded-lg flex-shrink-0">
            <p className="text-sm text-muted-foreground">{map.description}</p>
          </div>
          
          <div className="flex-1 relative min-h-0">
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-muted/50 rounded-lg z-10">
                <div className="text-center space-y-2">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                  <p className="text-sm text-muted-foreground">Загрузка карты...</p>
                </div>
              </div>
            )}

            {error ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center space-y-4 p-6">
                  <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                  <div className="flex gap-2 justify-center">
                    <Button variant="outline" onClick={() => {
                      setError(null);
                      setIsLoading(true);
                    }}>
                      Повторить
                    </Button>
                    <Button onClick={handleOpenInNewTab}>
                      Открыть в новой вкладке
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <iframe
                src={mapUrl}
                className="w-full h-full border rounded-lg"
                title={map.name}
                onLoad={handleIframeLoad}
                onError={handleIframeError}
                sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
              />
            )}
          </div>

          <div className="flex gap-2 justify-between items-center flex-shrink-0">
            <div className="text-xs text-muted-foreground">
              <span className="font-medium">Ссылка:</span>
              <code className="ml-1 px-1 py-0.5 bg-muted rounded text-xs">
                {map.url}
              </code>
            </div>
            
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleOpenInNewTab}>
                Открыть в новой вкладке
              </Button>
              <Button variant="outline" onClick={onClose}>
                Закрыть
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
