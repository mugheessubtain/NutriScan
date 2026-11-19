import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import Header from "@/components/Header";
import "./globals.css";

const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-plex-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-plex-mono" });

export const metadata: Metadata = {
  title: "NutriScan",
  description: "Ingredient label analysis: processing level, health tier and healthier alternatives.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen antialiased">
        <div className="backdrop" aria-hidden />
        <Header />
        <main className="mx-auto max-w-6xl px-4 pb-20 pt-6 sm:px-6">{children}</main>
        <footer className="border-t border-line/70">
          <div className="mx-auto max-w-6xl space-y-1 px-4 py-6 text-xs leading-relaxed text-faint sm:px-6">
            <p>
              NutriScan uses established classifications (NOVA, Nutri-Score) and published ratings. It is not medical
              advice. Always read the actual label.
            </p>
            <p>
              Contains data from{" "}
              <a className="underline underline-offset-2 hover:text-muted" href="https://world.openfoodfacts.org">
                Open Food Facts
              </a>
              , available under the{" "}
              <a className="underline underline-offset-2 hover:text-muted" href="https://opendatacommons.org/licenses/odbl/1-0/">
                Open Database License
              </a>
              . Product ingredient data also from{" "}
              <a className="underline underline-offset-2 hover:text-muted" href="https://fdc.nal.usda.gov">
                USDA FoodData Central
              </a>{" "}
              (public domain).
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
