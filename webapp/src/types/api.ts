export interface MapItem {
  description: string;
  name: string;
  url: string;
}

export interface MapsData {
  analysis: MapItem[];
  congestion: MapItem[];
  heatmaps: MapItem[];
}

export interface ApiResponse {
  maps: MapsData;
  message: string;
  total_maps: number;
  version: string;
}

export interface ApiError {
  error: string;
  message: string;
}
