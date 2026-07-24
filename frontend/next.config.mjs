/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    const backendBaseUrl = process.env.SEATFLOW_JAVA_INTERNAL_BASE_URL ?? "http://127.0.0.1:8080";
    return [
      {
        source: "/backend/:path*",
        destination: `${backendBaseUrl}/:path*`
      }
    ];
  }
};

export default nextConfig;
