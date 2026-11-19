import type { NextConfig } from "next";

// The Python API (runtime/api/server.py) runs on port 8000; the browser only talks to Next.js,
// which forwards /api/* there. Analyses on a low-RAM laptop can take ~20 s, hence the long proxy timeout.
const API_URL = process.env.NUTRISCAN_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  experimental: { proxyTimeout: 180_000 },
};

export default nextConfig;
