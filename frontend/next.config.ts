/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/proxy/:path*",                // browser calls /proxy/…
        destination: "http://localhost:8000/:path*", // hits FastAPI /… 
      },
    ];
  },
};

module.exports = nextConfig;
