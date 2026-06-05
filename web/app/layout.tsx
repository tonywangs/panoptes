import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Sidebar, MobileNav } from "@/components/Nav";
import { ScrollProgress } from "@/components/Motion";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PANOPTES · uncertainty-aware LLM evaluation",
  description:
    "Calibrated posteriors over LLM-judge scores. Aleatoric and epistemic decomposition, conformal prediction with finite-sample guarantees, Thompson-sampling jury routing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      data-scroll-behavior="smooth"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col md:flex-row">
        <ScrollProgress />
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <MobileNav />
          <main className="flex-1 p-6 md:p-10 max-w-6xl w-full mx-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
