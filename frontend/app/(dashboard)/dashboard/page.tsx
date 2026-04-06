"use client";

import Link from "next/link";
import {
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import { Button } from "@/components/ui/button";
import {
  Mic,
  Activity,
  Brain,
  Shield,
  ArrowRight,
  AudioWaveform,
  BarChart3,
  Stethoscope,
} from "lucide-react";

const modelComparisonData = [
  { model: "LogReg", score: 0.7346 },
  { model: "SVM", score: 0.7610 },
  { model: "RF", score: 0.7557 },
  { model: "GB", score: 0.7435 },
  { model: "AVHFM", score: 0.7645 },
];

const featureDistributionData = [
  { name: "Pitch", value: 16, color: "#14b8a6" },
  { name: "MFCC", value: 28, color: "#22c55e" },
  { name: "Perturbation", value: 18, color: "#eab308" },
  { name: "Spectral", value: 14, color: "#f97316" },
  { name: "Derived", value: 24, color: "#ef4444" },
];

const featureRadarData = [
  { feature: "Pitch", value: 82 },
  { feature: "MFCC", value: 91 },
  { feature: "Jitter", value: 76 },
  { feature: "Shimmer", value: 79 },
  { feature: "HNR", value: 74 },
  { feature: "AVHI", value: 88 },
];

const workflowSteps = [
  {
    id: 1,
    title: "Voice Input",
    description: "User uploads or records a short WAV voice sample.",
  },
  {
    id: 2,
    title: "Feature Extraction",
    description:
      "Pitch, MFCC, jitter, shimmer, HNR, and derived interaction features are extracted.",
  },
  {
    id: 3,
    title: "AVHFM Screening",
    description:
      "The custom hybrid ML model classifies the sample as healthy-like or irregular.",
  },
  {
    id: 4,
    title: "Result Output",
    description:
      "The system displays confidence, vocal metrics, insights, and screening guidance.",
  },
];

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">
            Vocare Project Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Overview of the AI-based vocal health early screening system
          </p>
        </div>
        <Link href="/analyze">
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
            <Mic className="w-4 h-4 mr-2" />
            Start New Analysis
          </Button>
        </Link>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            title: "Model Accuracy",
            value: "73.6",
            unit: "%",
            change: "Custom AVHFM model",
            icon: Activity,
            color: "text-primary",
          },
          {
            title: "F1-Score",
            value: "76.45",
            unit: "%",
            change: "Best overall screening balance",
            icon: Brain,
            color: "text-chart-2",
          },
          {
            title: "ROC-AUC",
            value: "81.87",
            unit: "%",
            change: "Strong separation capability",
            icon: BarChart3,
            color: "text-chart-4",
          },
          {
            title: "Screening Type",
            value: "Binary",
            unit: "",
            change: "Healthy-like vs Irregular",
            icon: Shield,
            color: "text-chart-5",
          },
        ].map((stat, index) => (
          <Card key={index} className="bg-card border-border">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{stat.title}</p>
                  <p className="text-2xl font-bold mt-1">
                    {stat.value}
                    <span className="text-lg text-muted-foreground">
                      {stat.unit}
                    </span>
                  </p>
                  <p className="text-xs text-primary mt-1">{stat.change}</p>
                </div>
                <div className={`p-3 rounded-lg bg-secondary ${stat.color}`}>
                  <stat.icon className="w-6 h-6" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Comparison */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle>Model Comparison</CardTitle>
            <CardDescription>
              F1-score comparison between baseline models and the AVHFM model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                score: {
                  label: "F1-Score",
                  color: "hsl(var(--chart-1))",
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelComparisonData}>
                  <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="model"
                    stroke="#9ca3af"
                    tick={{ fill: "#9ca3af", fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#9ca3af"
                    tick={{ fill: "#9ca3af", fontSize: 12 }}
                    domain={[0.7, 0.78]}
                  />
                  <ChartTooltip
                    cursor={false}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="rounded-lg border border-border bg-card p-2 shadow-md">
                            <p className="text-sm font-medium">
                              {payload[0].payload.model}
                            </p>
                            <p className="text-sm text-primary">
                              F1-Score: {payload[0].value}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar
                    dataKey="score"
                    fill="#14b8a6"
                    radius={[8, 8, 0, 0]}
                    
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Feature Radar */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle>Acoustic Feature Coverage</CardTitle>
            <CardDescription>
              Key voice feature groups used in the screening pipeline
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                value: {
                  label: "Coverage",
                  color: "hsl(var(--chart-2))",
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={featureRadarData}>
                  <PolarGrid stroke="#374151"/>
                  <PolarAngleAxis
                    dataKey="feature"
                    tick={{ fill: "#9ca3af", fontSize: 12 }}
                  />

                  <PolarRadiusAxis
                    tick={{ fill: "#6b7280", fontSize: 10 }}
                  />
                  <Radar
                    name="Coverage"
                    dataKey="value"
                    stroke="#14b8a6"
                    fill="#14b8a6"
                    fillOpacity={0.5}
                  />
                  <ChartTooltip />
                </RadarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Feature Distribution */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle>Feature Distribution</CardTitle>
            <CardDescription>
              Major categories of extracted acoustic features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[220px] flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={featureDistributionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {featureDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <ChartTooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="flex flex-wrap justify-center gap-4 mt-4">
              {featureDistributionData.map((item, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-muted-foreground">
                    {item.name}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Project Workflow */}
        <Card className="bg-card border-border lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>System Workflow</CardTitle>
              <CardDescription>
                How Vocare processes a voice sample from input to result
              </CardDescription>
            </div>
            <Link href="/analyze">
              <Button variant="ghost" size="sm" className="text-primary">
                Go to Analysis
                <ArrowRight className="w-4 h-4 ml-1" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {workflowSteps.map((step) => (
                <div
                  key={step.id}
                  className="flex items-start justify-between p-4 rounded-lg bg-secondary gap-4"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-full bg-primary/15 text-primary flex items-center justify-center font-semibold">
                      {step.id}
                    </div>
                    <div>
                      <p className="font-medium">{step.title}</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {step.description}
                      </p>
                    </div>
                  </div>
                  <div className="hidden sm:flex w-10 h-10 rounded-full bg-background items-center justify-center text-primary">
                    {step.id === 1 && <Mic className="w-5 h-5" />}
                    {step.id === 2 && <AudioWaveform className="w-5 h-5" />}
                    {step.id === 3 && <Brain className="w-5 h-5" />}
                    {step.id === 4 && <Stethoscope className="w-5 h-5" />}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Note */}
      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold">Clinical Scope Note</h3>
              <p className="text-sm text-muted-foreground mt-1 max-w-3xl">
                Vocare is designed as an AI-based early screening system for
                possible vocal irregularities. It does not provide confirmed
                disease diagnosis and should be used as a support tool before
                specialist consultation.
              </p>
            </div>
            <Link href="/analyze">
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                Analyze Voice Sample
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}