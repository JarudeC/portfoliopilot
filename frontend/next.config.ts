/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      {
        source: '/api/forecast/:path*',
        destination: `${backendUrl}/forecast/:path*`, // FastAPI
      },
      {
        source: "/proxy/:path*",                // browser calls /proxy/…
        destination: `${backendUrl}/:path*`, // hits FastAPI /…
      },
    ];
  },
};

module.exports = nextConfig;
