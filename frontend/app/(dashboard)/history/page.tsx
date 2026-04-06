"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  History as HistoryIcon,
  CheckCircle,
  AlertTriangle,
  Activity,
  Trash2,
  Mic,
  CalendarDays,
  FileAudio,
} from "lucide-react";
import Link from "next/link";

type HistoryItem = {
  id: number;
  date: string;
  time: string;
  result: string;
  score: number;
  status: "healthy" | "warning" | "risk";
  duration: string;
  fileName: string;
  metrics: {
    pitch: number;
    clarity: number;
    strain: number;
    stability: number;
  };
  raw_metrics?: {
    pitch_mean_hz?: number;
    pitch_std_dev?: number;
    jitter?: number;
    shimmer?: number;
    hnr?: number;
    avhi?: number;
  };
  risk_level?: string;
  audio_quality?: string;
  recommendations?: string[];
  insights?: string[];
  message?: string;
};

export default function HistoryPage() {
  const [historyData, setHistoryData] = useState<HistoryItem[]>([]);

  const loadHistory = () => {
    const saved = localStorage.getItem("vocare_history");
    const parsed = saved ? JSON.parse(saved) : [];
    setHistoryData(parsed);
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const totalAnalyses = historyData.length;

  const healthyCount = useMemo(
    () => historyData.filter((item) => item.status === "healthy").length,
    [historyData]
  );

  const irregularCount = useMemo(
    () =>
      historyData.filter(
        (item) => item.status === "warning" || item.status === "risk"
      ).length,
    [historyData]
  );

  const averageScore = useMemo(() => {
    if (!historyData.length) return 0;
    const total = historyData.reduce((sum, item) => sum + item.score, 0);
    return Math.round(total / historyData.length);
  }, [historyData]);

  const clearHistory = () => {
    localStorage.removeItem("vocare_history");
    setHistoryData([]);
  };

  const deleteOne = (id: number) => {
    const updated = historyData.filter((item) => item.id !== id);
    localStorage.setItem("vocare_history", JSON.stringify(updated));
    setHistoryData(updated);
  };

  const getStatusIcon = (status: HistoryItem["status"]) => {
    if (status === "healthy") {
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    }
    return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
  };

  const getStatusBadge = (status: HistoryItem["status"]) => {
    if (status === "healthy") {
      return "bg-green-500/10 text-green-400 border-green-500/20";
    }
    if (status === "warning") {
      return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
    }
    return "bg-red-500/10 text-red-400 border-red-500/20";
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">Analysis History</h1>
          <p className="text-muted-foreground mt-1">
            View previous voice screening results stored in this browser.
          </p>
        </div>

        <div className="flex gap-3">
          <Link href="/analyze">
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
              <Mic className="w-4 h-4 mr-2" />
              New Analysis
            </Button>
          </Link>

          {historyData.length > 0 && (
            <Button
              variant="outline"
              onClick={clearHistory}
              className="border-border bg-transparent"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear History
            </Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">Total Analyses</p>
            <div className="flex items-center justify-between mt-3">
              <span className="text-3xl font-bold">{totalAnalyses}</span>
              <HistoryIcon className="w-6 h-6 text-primary" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">Healthy-like Results</p>
            <div className="flex items-center justify-between mt-3">
              <span className="text-3xl font-bold">{healthyCount}</span>
              <CheckCircle className="w-6 h-6 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">Irregular Results</p>
            <div className="flex items-center justify-between mt-3">
              <span className="text-3xl font-bold">{irregularCount}</span>
              <AlertTriangle className="w-6 h-6 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">Average Score</p>
            <div className="flex items-center justify-between mt-3">
              <span className="text-3xl font-bold">{averageScore}</span>
              <Activity className="w-6 h-6 text-primary" />
            </div>
          </CardContent>
        </Card>
      </div>

      {historyData.length === 0 ? (
        <Card className="bg-card border-border">
          <CardContent className="p-10 text-center">
            <div className="w-16 h-16 rounded-full bg-secondary mx-auto flex items-center justify-center mb-4">
              <HistoryIcon className="w-8 h-8 text-muted-foreground" />
            </div>
            <h2 className="text-xl font-semibold">No analysis history yet</h2>
            <p className="text-muted-foreground mt-2 max-w-xl mx-auto">
              Your completed voice screenings will appear here after you analyze
              a sample.
            </p>
            <Link href="/analyze">
              <Button className="mt-6 bg-primary text-primary-foreground hover:bg-primary/90">
                Start First Analysis
              </Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {historyData.map((item) => (
            <Card key={item.id} className="bg-card border-border">
              <CardContent className="p-6">
                <div className="flex flex-col xl:flex-row xl:items-start xl:justify-between gap-5">
                  <div className="flex-1 space-y-4">
                    <div className="flex flex-col md:flex-row md:items-center gap-3 md:gap-4">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(item.status)}
                        <h2 className="text-lg font-semibold">{item.result}</h2>
                      </div>

                      <div
                        className={`w-fit px-3 py-1 rounded-full border text-xs font-medium ${getStatusBadge(
                          item.status
                        )}`}
                      >
                        {item.risk_level || item.status.toUpperCase()}
                      </div>
                    </div>

                    <p className="text-sm text-muted-foreground">
                      {item.message || "AI-based vocal screening result."}
                    </p>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="rounded-lg bg-secondary/40 p-3">
                        <p className="text-xs text-muted-foreground">Score</p>
                        <p className="text-lg font-semibold">{item.score}/100</p>
                      </div>

                      <div className="rounded-lg bg-secondary/40 p-3">
                        <p className="text-xs text-muted-foreground">Duration</p>
                        <p className="text-lg font-semibold">{item.duration}</p>
                      </div>

                      <div className="rounded-lg bg-secondary/40 p-3">
                        <p className="text-xs text-muted-foreground">Pitch</p>
                        <p className="text-lg font-semibold">
                          {item.metrics.pitch}
                        </p>
                      </div>

                      <div className="rounded-lg bg-secondary/40 p-3">
                        <p className="text-xs text-muted-foreground">
                          Stability
                        </p>
                        <p className="text-lg font-semibold">
                          {item.metrics.stability}
                        </p>
                      </div>
                    </div>

                    <div className="flex flex-col md:flex-row md:items-center gap-3 text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <CalendarDays className="w-4 h-4" />
                        <span>
                          {item.date} at {item.time}
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <FileAudio className="w-4 h-4" />
                        <span className="break-all">{item.fileName}</span>
                      </div>

                      {item.audio_quality && (
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4" />
                          <span>Audio: {item.audio_quality}</span>
                        </div>
                      )}
                    </div>

                    {!!item.insights?.length && (
                      <div className="rounded-xl bg-secondary/30 p-4">
                        <p className="font-medium mb-2">Insights</p>
                        <ul className="space-y-1 text-sm text-muted-foreground">
                          {item.insights.slice(0, 3).map((insight, index) => (
                            <li key={index}>• {insight}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  <div className="flex xl:flex-col gap-3">
                    <Button
                      variant="outline"
                      onClick={() => deleteOne(item.id)}
                      className="border-border bg-transparent"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle>History Note</CardTitle>
          <CardDescription>
            How this page currently stores data
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            This history is currently stored in your browser using localStorage.
          </p>
          <p>
            That means it stays on this device and browser, but it is not shared
            across other devices or accounts.
          </p>
          <p>
            Later, you can upgrade this to a backend database if you want real
            user accounts and permanent cloud history.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}