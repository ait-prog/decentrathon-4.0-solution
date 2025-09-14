## Backend Integration

The app connects to a backend server running on `http://192.168.0.102:8081/` that provides:

- **Analysis Maps**: Traffic clustering and movement analysis
- **Congestion Maps**: Traffic bottleneck and congestion visualizations  
- **Heatmaps**: Density and traffic flow visualizations

## Getting Started

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## Prerequisites

- Node.js 18+ 
- Backend server running on `http://192.168.0.102:8081/`

## Project Structure

```
src/
├── app/                 # Next.js app directory
├── components/          # React components
│   ├── ui/             # shadcn/ui components
│   ├── MapsDashboard.tsx
│   ├── MapCard.tsx
│   ├── MapViewer.tsx
│   └── ConnectionStatus.tsx
├── lib/                # Utility functions
│   ├── api.ts          # API service
│   └── utils.ts        # shadcn/ui utilities
└── types/              # TypeScript type definitions
    └── api.ts          # API response types
```


## API Endpoints

The app expects the backend to provide a JSON response at the root endpoint (`/`) with the following structure:

```json
{
  "maps": {
    "analysis": [...],
    "congestion": [...], 
    "heatmaps": [...]
  },
  "message": "Description",
  "total_maps": 10,
  "version": "1.0.0"
}
```
