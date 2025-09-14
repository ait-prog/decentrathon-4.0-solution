import { MapsDashboard } from '@/components/MapsDashboard';

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <MapsDashboard />
      </div>
    </div>
  );
}
