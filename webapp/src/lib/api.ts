import { ApiResponse, ApiError, MapsData, MapItem } from '@/types/api';

const BACKEND_URL = 'http://192.168.0.102:8081';

export class ApiService {
  private static async fetchWithErrorHandling<T>(
    url: string,
    options?: RequestInit
  ): Promise<T> {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const errorData: ApiError = await response.json().catch(() => ({
          error: 'Network Error',
          message: `HTTP ${response.status}: ${response.statusText}`,
        }));
        throw new Error(errorData.message || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('An unexpected error occurred');
    }
  }

  static async getMaps(category?: string): Promise<ApiResponse> {
    const url = `${BACKEND_URL}${category ? `/maps/${category}` : '/'}`;
    const response = await this.fetchWithErrorHandling<ApiResponse>(url);
    
    // Transform the maps object into the expected format
    if (response.maps) {
      const mapsData: MapsData = {
        analysis: response.maps.analysis || [],
        congestion: response.maps.congestion || [],
        heatmaps: response.maps.heatmaps || []
      };
      
      // Ensure all URLs are absolute
      (Object.entries(mapsData) as [keyof MapsData, MapItem[]][]).forEach(([_, categoryMaps]) => {
        categoryMaps.forEach((map: MapItem) => {
          if (!map.url.startsWith('http')) {
            map.url = this.getMapUrl(map.url);
          }
        });
      });
      
      return {
        ...response,
        maps: mapsData
      };
    }
    
    // Return empty maps if none found
    return {
      ...response,
      maps: {
        analysis: [],
        congestion: [],
        heatmaps: []
      }
    };
  }

  static getMapUrl(mapUrl: string): string {
    return mapUrl.startsWith('http') ? mapUrl : `${BACKEND_URL}${mapUrl}`;
  }

  static async testConnection(): Promise<boolean> {
    try {
      await this.getMaps();
      return true;
    } catch {
      return false;
    }
  }
}
