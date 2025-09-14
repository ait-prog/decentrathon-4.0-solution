'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { MapItem } from '@/types/api';
import { MapViewer } from './MapViewer';

interface MapCardProps {
  map: MapItem;
  category: string;
}

export function MapCard({ map, category }: MapCardProps) {
  const [isViewerOpen, setIsViewerOpen] = useState(false);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'analysis':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'congestion':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
      case 'heatmaps':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getCategoryName = (category: string) => {
    switch (category) {
      case 'analysis':
        return 'Анализ';
      case 'congestion':
        return 'Пробки';
      case 'heatmaps':
        return 'Тепловые карты';
      default:
        return category;
    }
  };

  return (
    <>
      <Card className="h-full flex flex-col">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="text-lg leading-tight">{map.name}</CardTitle>
            <Badge className={getCategoryColor(category)}>
              {getCategoryName(category)}
            </Badge>
          </div>
          <CardDescription className="text-sm">
            {map.description}
          </CardDescription>
        </CardHeader>
        
        <CardContent className="pt-0 flex-1 flex flex-col justify-end">
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              <span className="font-medium">Путь:</span>
              <code className="ml-1 px-1 py-0.5 bg-muted rounded text-xs">
                {map.url}
              </code>
            </div>
            
            <Button 
              onClick={() => setIsViewerOpen(true)}
              className="w-full"
            >
              Просмотреть карту
            </Button>
          </div>
        </CardContent>
      </Card>

      <MapViewer
        map={map}
        isOpen={isViewerOpen}
        onClose={() => setIsViewerOpen(false)}
      />
    </>
  );
}
