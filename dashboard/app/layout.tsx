
import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Octagon AI",
    description: "Advanced AI-Powered UFC Predictions",
    icons: {
        icon: "/logo.png",
    },
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="dark">
            <body className="antialiased min-h-screen bg-black text-white selection:bg-red-900 selection:text-white">
                {children}
                <Analytics />
            </body>
        </html >
    );
}
