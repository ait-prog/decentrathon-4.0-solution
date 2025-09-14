'use client';

import { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ApiService } from '@/lib/api';

export function ConnectionStatus() {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(false);

  const checkConnection = async () => {
    setIsChecking(true);
    try {
      const connected = await ApiService.testConnection();
      setIsConnected(connected);
    } catch {
      setIsConnected(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkConnection();
  }, []);

  const getStatusBadge = () => {
    if (isConnected === null) {
      return (
        <Badge variant="secondary" className="animate-pulse">
          Checking...
        </Badge>
      );
    }

    if (isConnected) {
      return (
        <Badge variant="default" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
          Connected
        </Badge>
      );
    }

    return (
      <Badge variant="destructive">
        Disconnected
      </Badge>
    );
  };

  return (
    <div className="flex items-center gap-2">
      {getStatusBadge()}
      <Button
        variant="outline"
        size="sm"
        onClick={checkConnection}
        disabled={isChecking}
      >
        {isChecking ? 'Checking...' : 'Retry'}
      </Button>
    </div>
  );
}
