"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Mic,
  Brain,
  Shield,
  ArrowRight,
  Waves,
  ChevronRight,
  Stethoscope,
} from "lucide-react";

export default function LandingPage() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const waveformHeights = [
    28, 34, 42, 55, 57, 54, 61, 59, 48, 52,
    39, 16, 20, 8, 6, 9, 7, 5, 4, 22,
    15, 24, 41, 57, 58, 50, 56, 51, 48, 57,
    47, 42, 27, 8, 5, 4, 4, 6, 17, 23,
  ];

  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-primary/10 rounded-full blur-3xl animate-pulse delay-1000" />
      </div>

      <nav className="relative z-10 flex items-center justify-between px-6 py-4 md:px-12 lg:px-20">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
            <Waves className="w-6 h-6 text-primary-foreground" />
          </div>
          <span className="text-xl font-bold">Vocare</span>
        </div>

        <div className="hidden md:flex items-center gap-8">
          <Link
            href="#features"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Features
          </Link>
          <Link
            href="#technology"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Technology
          </Link>
          <Link
            href="#about"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            About
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <Link href="/analyze">
            <Button variant="ghost" className="text-foreground">
              Analyze Now
            </Button>
          </Link>
          <Link href="/analyze">
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
              Start Screening
            </Button>
          </Link>
        </div>
      </nav>

      <section className="relative z-10 px-6 pt-20 pb-32 md:px-12 lg:px-20">
        <div
          className={`max-w-5xl mx-auto text-center transition-all duration-1000 ${
            isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
          }`}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary border border-border mb-8">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm text-muted-foreground">
              AI-Powered Early Voice Screening
            </span>
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight text-balance">
            Voice Health
            <br />
            <span className="text-primary">Made Simple</span>
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-10 text-pretty">
            Early screening of vocal irregularities using acoustic feature extraction
            and a custom hybrid machine learning model. Upload your voice sample
            and receive fast AI-based insights into possible vocal health changes.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/analyze">
              <Button
                size="lg"
                className="bg-primary text-primary-foreground hover:bg-primary/90 px-8 py-6 text-lg group"
              >
                Start Analysis
                <ArrowRight className="ml-2 w-5 h-5 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
            <Link href="#features">
              <Button
                size="lg"
                variant="outline"
                className="px-8 py-6 text-lg border-border text-foreground hover:bg-secondary bg-transparent"
              >
                Learn More
              </Button>
            </Link>
          </div>
        </div>

        <div
          className={`mt-20 flex items-end justify-center gap-1 transition-all duration-1000 delay-500 ${
            isVisible ? "opacity-100" : "opacity-0"
          }`}
        >
          {waveformHeights.map((height, i) => (
            <div
              key={i}
              className="w-1 bg-primary/60 rounded-full animate-pulse"
              style={{
                height: `${height}px`,
                animationDelay: `${i * 50}ms`,
                animationDuration: "1s",
              }}
            />
          ))}
        </div>
      </section>

      <section
        id="features"
        className="relative z-10 px-6 py-24 md:px-12 lg:px-20 bg-secondary/50"
      >
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Advanced Voice Analysis
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Vocare combines acoustic analysis and custom machine learning
              to support accessible early screening for vocal health changes.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Mic,
                title: "Voice Upload",
                description:
                  "Upload a short WAV voice sample for quick and simple vocal screening.",
              },
              {
                icon: Brain,
                title: "AVHFM Hybrid Model",
                description:
                  "Our custom AVHFM model analyzes pitch, MFCC, jitter, shimmer, HNR, and derived interaction features.",
              },
              {
                icon: Activity,
                title: "Smart Metrics",
                description:
                  "View vocal indicators such as pitch variation, perturbation, clarity, stability, and AVHI-related features.",
              },
              {
                icon: Shield,
                title: "Screening Support",
                description:
                  "Designed as an AI-assisted early screening tool and not as a replacement for clinical diagnosis.",
              },
            ].map((feature, index) => (
              <div
                key={index}
                className="group p-6 rounded-xl bg-card border border-border hover:border-primary/50 transition-all duration-300 hover:-translate-y-1"
              >
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section
        id="technology"
        className="relative z-10 px-6 py-24 md:px-12 lg:px-20"
      >
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                Powered by Acoustic Intelligence
              </h2>
              <p className="text-muted-foreground mb-8">
                The system uses voice signal preprocessing, acoustic feature extraction,
                and a custom AVHFM hybrid machine learning pipeline for binary early
                screening of healthy-like versus irregular vocal patterns.
              </p>

              <div className="space-y-4">
                {[
                  "MFCC Feature Extraction",
                  "Pitch, Jitter, Shimmer, and HNR Analysis",
                  "AVHI and Derived Interaction Features",
                  "AVHFM Hybrid ML Classification",
                ].map((item, index) => (
                  <div key={index} className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
                      <ChevronRight className="w-4 h-4 text-primary" />
                    </div>
                    <span>{item}</span>
                  </div>
                ))}
              </div>

              <Link href="/analyze">
                <Button className="mt-8 bg-primary text-primary-foreground hover:bg-primary/90">
                  Run Analysis
                  <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
            </div>

            <div className="relative">
              <div className="aspect-square rounded-2xl bg-card border border-border p-8 relative overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-48 h-48">
                    {[1, 2, 3, 4].map((ring) => (
                      <div
                        key={ring}
                        className="absolute inset-0 rounded-full border border-primary/30 animate-ping"
                        style={{
                          animationDelay: `${ring * 0.5}s`,
                          animationDuration: "3s",
                          transform: `scale(${1 + ring * 0.3})`,
                        }}
                      />
                    ))}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center">
                        <Mic className="w-10 h-10 text-primary" />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="absolute bottom-4 left-4 right-4 grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-secondary/80 backdrop-blur px-3 py-2 rounded-lg">
                    <div className="text-muted-foreground">Accuracy</div>
                    <div className="text-lg font-bold text-primary">73.6%</div>
                  </div>
                  <div className="bg-secondary/80 backdrop-blur px-3 py-2 rounded-lg">
                    <div className="text-muted-foreground">F1-Score</div>
                    <div className="text-lg font-bold text-primary">76.45%</div>
                  </div>
                  <div className="bg-secondary/80 backdrop-blur px-3 py-2 rounded-lg">
                    <div className="text-muted-foreground">ROC-AUC</div>
                    <div className="text-lg font-bold text-primary">81.87%</div>
                  </div>
                  <div className="bg-secondary/80 backdrop-blur px-3 py-2 rounded-lg">
                    <div className="text-muted-foreground">Model Type</div>
                    <div className="text-lg font-bold text-primary">AVHFM</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section
        id="about"
        className="relative z-10 px-6 py-24 md:px-12 lg:px-20"
      >
        <div className="max-w-4xl mx-auto text-center">
          <div className="p-12 rounded-2xl bg-card border border-border relative overflow-hidden">
            <div className="absolute inset-0 bg-primary/5" />
            <div className="relative z-10">
              <div className="flex justify-center mb-4">
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
                  <Stethoscope className="w-7 h-7 text-primary" />
                </div>
              </div>

              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Start Your Voice Screening
              </h2>

              <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
                Vocare is designed for users with high voice usage such as teachers,
                singers, call-center professionals, public speakers, and others who
                want fast AI-assisted screening for possible vocal irregularities.
              </p>

              <p className="text-sm text-muted-foreground mb-8 max-w-2xl mx-auto">
                This system provides AI-based early screening support and does not
                replace medical diagnosis. For persistent symptoms such as hoarseness,
                vocal fatigue, strain, or pain, consultation with an ENT specialist
                or speech expert is recommended.
              </p>

              <Link href="/analyze">
                <Button
                  size="lg"
                  className="bg-primary text-primary-foreground hover:bg-primary/90 px-8"
                >
                  Start Free Screening
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="relative z-10 px-6 py-12 md:px-12 lg:px-20 border-t border-border">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Waves className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-semibold">Vocare</span>
          </div>

          <p className="text-sm text-muted-foreground">
            2026 Vocare. AI-based vocal health early screening.
          </p>

          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <span className="hover:text-foreground transition-colors cursor-default">
              Privacy
            </span>
            <span className="hover:text-foreground transition-colors cursor-default">
              Terms
            </span>
            <span className="hover:text-foreground transition-colors cursor-default">
              Contact
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}