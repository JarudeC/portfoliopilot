import React from "react";

export default function Footer() {
  return (
    <footer className="bg-[#14273F] py-10 text-center text-gray-400">
      <div className="max-w-7xl mx-auto px-6">
        <p className="mb-4">
            PortfolioPilot by Jared Chan
        </p>
        <div className="flex justify-center gap-6 text-sm">
          <a href="https://github.com/your-repo" className="hover:text-white">
            GitHub
          </a>
          <a href="#hero" className="hover:text-white">
            Back to Top
          </a>
        </div>
      </div>
    </footer>
  );
}
