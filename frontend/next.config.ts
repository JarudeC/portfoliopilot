/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/forecast/:path*',
        destination: 'http://localhost:8000/forecast/:path*', // FastAPI
      },
      {
        source: "/proxy/:path*",                // browser calls /proxy/…
        destination: "http://localhost:8000/:path*", // hits FastAPI /… 
      },
    ];
  },
};

module.exports = nextConfig;
