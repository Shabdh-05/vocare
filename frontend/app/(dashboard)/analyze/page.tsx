"use client";

import React, { useCallback, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Mic,
  Upload,
  Square,
  CheckCircle,
  AlertTriangle,
  Activity,
  Waves,
  Volume2,
  Clock,
  RefreshCw,
  FileAudio,
  Shield,
  Brain,
  Stethoscope,
  AudioWaveform,
} from "lucide-react";

type AnalysisResult = {
  health_score: any;
  status: "healthy" | "warning" | "risk";
  score: number;
  diagnosis: string;
  message?: string;
  risk_level?: string;
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
  recommendations: string[];
  quality_info?: {
    audio_quality_score?: number;
    audio_quality_label?: string;
    duration_sec?: number;
  };
  insights?: string[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:5000";

const guidelines = [
  "Upload a clear WAV voice sample.",
  "Recommended duration: 3 to 10 seconds.",
  "Record in a quiet environment with low background noise.",
  "Sustained vowels like 'aaa' or a short clear phrase work best.",
];

const featureList = [
  "Pitch mean and pitch variation",
  "Jitter and shimmer",
  "Harmonic-to-noise ratio (HNR)",
  "MFCC-based acoustic patterns",
  "AVHI and derived interaction features",
];

const workflowSteps = [
  {
    title: "Voice Input",
    desc: "Upload a WAV sample or use the prototype recording interface.",
    icon: Mic,
  },
  {
    title: "Feature Extraction",
    desc: "The system extracts pitch, MFCC, jitter, shimmer, HNR, and AVHI-related features.",
    icon: AudioWaveform,
  },
  {
    title: "AVHFM Screening",
    desc: "The custom hybrid ML model evaluates the sample for healthy-like or irregular vocal patterns.",
    icon: Brain,
  },
  {
    title: "Clinical Output",
    desc: "Vocare displays screening status, confidence, metrics, insights, and recommendations.",
    icon: Stethoscope,
  },
];

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeProgress, setAnalyzeProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleDrag = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (
        droppedFile.type.startsWith("audio/") ||
        droppedFile.name.toLowerCase().endsWith(".wav")
      ) {
        setFile(droppedFile);
        setResult(null);
        setError("");
      } else {
        setError("Please upload a valid WAV audio file.");
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (
        selectedFile.type.startsWith("audio/") ||
        selectedFile.name.toLowerCase().endsWith(".wav")
      ) {
        setFile(selectedFile);
        setResult(null);
        setError("");
      } else {
        setError("Please upload a valid WAV audio file.");
      }
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      // STOP recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const webmBlob = new Blob(audioChunksRef.current, {
            type: "audio/webm",
          });

          // 🔥 Convert WebM → WAV
          const arrayBuffer = await webmBlob.arrayBuffer();
          const audioContext = new AudioContext();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          const wavBlob = audioBufferToWav(audioBuffer);

          const recordedFile = new File([wavBlob], "recording.wav", {
            type: "audio/wav",
          });

          setFile(recordedFile);
        };

        mediaRecorder.start();
        setIsRecording(true);
        setFile(null);
        setResult(null);
      } catch (err) {
        console.error("Mic access error:", err);
        alert("Microphone access denied");
      }
    }
  };

  const saveAnalysisToHistory = (analysis: AnalysisResult, uploadedFile: File) => {
    const now = new Date();

    const historyItem = {
      id: Date.now(),
      date: now.toISOString().split("T")[0],
      time: now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      result: analysis.diagnosis,
      score: analysis.score,
      status: analysis.status,
      duration: analysis.quality_info?.duration_sec
        ? `${analysis.quality_info.duration_sec}s`
        : "-",
      fileName: uploadedFile.name,
      metrics: {
        pitch: analysis.metrics.pitch,
        clarity: analysis.metrics.clarity,
        strain: analysis.metrics.strain,
        stability: analysis.metrics.stability,
      },
      raw_metrics: analysis.raw_metrics || {},
      risk_level: analysis.risk_level || "",
      audio_quality: analysis.quality_info?.audio_quality_label || "",
      recommendations: analysis.recommendations || [],
      insights: analysis.insights || [],
      message: analysis.message || "",
    };

    const existingHistory = localStorage.getItem("vocare_history");
    const parsedHistory = existingHistory ? JSON.parse(existingHistory) : [];

    const updatedHistory = [historyItem, ...parsedHistory];
    localStorage.setItem("vocare_history", JSON.stringify(updatedHistory));
  };

  const analyzeVoice = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalyzeProgress(10);
    setError("");
    setResult(null);

    const progressInterval = setInterval(() => {
      setAnalyzeProgress((prev) => {
        if (prev >= 90) return 90;
        return prev + 10;
      });
    }, 250);

    try {
      const formData = new FormData();
      formData.append("audio_file", file);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Analysis failed");
      }

      setAnalyzeProgress(100);
      setResult(data);
      saveAnalysisToHistory(data, file);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Something went wrong during analysis.";
      setError(message);
    } finally {
      clearInterval(progressInterval);
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setResult(null);
    setAnalyzeProgress(0);
    setError("");
    setIsRecording(false);
  };

  const getStatusColor = () => {
    if (!result) return "bg-primary";
    if (result.status === "healthy") return "bg-green-500";
    if (result.status === "warning") return "bg-yellow-500";
    return "bg-red-500";
  };

  const getStatusBg = () => {
    if (!result) return "bg-primary/20";
    if (result.status === "healthy") return "bg-green-500/20";
    if (result.status === "warning") return "bg-yellow-500/20";
    return "bg-red-500/20";
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold">Voice Analysis</h1>
        <p className="text-muted-foreground mt-1 max-w-3xl">
          Upload a short WAV voice sample to perform AI-based early screening
          of vocal health patterns using acoustic feature extraction and the
          AVHFM machine learning model.
        </p>
      </div>

      {error && (
        <Card className="border-red-500/40 bg-red-500/10">
          <CardContent className="p-4 text-red-200">{error}</CardContent>
        </Card>
      )}

      {!result ? (
        <>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="w-5 h-5 text-primary" />
                  Upload Recording
                </CardTitle>
                <CardDescription>
                  Upload a WAV voice sample for acoustic feature extraction and
                  AVHFM-based screening.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                    dragActive
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".wav,audio/wav"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center">
                      <Upload className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-medium">Drag and drop your WAV file</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        or click to browse from your system
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                      className="border-border"
                    >
                      Choose File
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Mic className="w-5 h-5 text-primary" />
                  Record Voice
                </CardTitle>
                <CardDescription>
                  Live recording is currently in prototype mode. Upload a WAV
                  file for full ML-based screening output.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center gap-6 py-4">
                  <div className="relative">
                    <button
                      onClick={toggleRecording}
                      className={`w-24 h-24 rounded-full flex items-center justify-center transition-all ${
                        isRecording
                          ? "bg-destructive animate-pulse"
                          : "bg-primary hover:bg-primary/90"
                      }`}
                    >
                      {isRecording ? (
                        <Square className="w-10 h-10 text-destructive-foreground" />
                      ) : (
                        <Mic className="w-10 h-10 text-primary-foreground" />
                      )}
                    </button>
                    {isRecording && (
                      <div className="absolute inset-0 rounded-full border-4 border-destructive animate-ping" />
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {isRecording
                      ? "Recording... Click to stop"
                      : "Click to start recording"}
                  </p>
                  {isRecording && (
                    <div className="flex items-center gap-1">
                      {Array.from({ length: 20 }).map((_, i) => (
                        <div
                          key={i}
                          className="w-1 bg-destructive rounded-full animate-pulse"
                          style={{
                            height: `${10 + Math.random() * 20}px`,
                            animationDelay: `${i * 50}ms`,
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileAudio className="w-5 h-5 text-primary" />
                  Sample Guidelines
                </CardTitle>
                <CardDescription>
                  Best practices for a reliable screening sample
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {guidelines.map((item, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <span className="mt-1 text-primary">•</span>
                      <span className="text-sm text-muted-foreground">
                        {item}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary" />
                  Acoustic Features Used
                </CardTitle>
                <CardDescription>
                  Main indicators extracted from the voice signal
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {featureList.map((item, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <span className="mt-1 text-primary">•</span>
                      <span className="text-sm text-muted-foreground">
                        {item}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-primary" />
                  Clinical Scope
                </CardTitle>
                <CardDescription>
                  Important usage information for this screening system
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Vocare is an AI-based early screening tool for possible vocal
                  irregularities.
                </p>
                <p>
                  It does not provide confirmed medical diagnosis or disease
                  verification.
                </p>
                <p>
                  Persistent symptoms such as hoarseness, pain, strain, or vocal
                  fatigue should be evaluated by an ENT specialist or speech
                  expert.
                </p>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>How Vocare Works</CardTitle>
              <CardDescription>
                End-to-end processing pipeline for voice screening
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                {workflowSteps.map((step, index) => (
                  <div
                    key={index}
                    className="rounded-xl border border-border bg-secondary/40 p-4"
                  >
                    <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                      <step.icon className="w-6 h-6 text-primary" />
                    </div>
                    <h3 className="font-semibold">{step.title}</h3>
                    <p className="text-sm text-muted-foreground mt-2">
                      {step.desc}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {file && (
            <Card className="bg-card border-border">
              <CardContent className="p-6">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Volume2 className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <Button
                      variant="outline"
                      onClick={resetAnalysis}
                      className="border-border bg-transparent"
                    >
                      Remove
                    </Button>
                    <Button
                      onClick={analyzeVoice}
                      disabled={isAnalyzing}
                      className="bg-primary text-primary-foreground hover:bg-primary/90"
                    >
                      {isAnalyzing ? (
                        <div className="flex items-center gap-2">
                          <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                          Analyzing...
                        </div>
                      ) : (
                        <>
                          <Activity className="w-4 h-4 mr-2" />
                          Analyze Voice
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {isAnalyzing && (
                  <div className="mt-6 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        Extracting audio features and evaluating AVHFM model...
                      </span>
                      <span className="text-primary">{analyzeProgress}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${analyzeProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </>
      ) : (
        <div className="space-y-6">
          <Card className="bg-card border-border overflow-hidden">
            <div className={`h-2 ${getStatusColor()}`} />
            <CardContent className="p-8">
              <div className="flex flex-col xl:flex-row items-center xl:items-start gap-8">
                <div
                  className={`w-32 h-32 rounded-full flex items-center justify-center ${getStatusBg()}`}
                >
                  {result.status === "healthy" ? (
                    <CheckCircle
                      className="w-16 h-16 text-green-500"
                      strokeWidth={1.5}
                    />
                  ) : (
                    <AlertTriangle
                      className={`w-16 h-16 ${
                        result.status === "warning"
                          ? "text-yellow-500"
                          : "text-red-500"
                      }`}
                      strokeWidth={1.5}
                    />
                  )}
                </div>

                <div className="text-center xl:text-left flex-1">
                  <p className="text-sm text-muted-foreground uppercase tracking-wide">
                    Screening Result
                  </p>
                  <h2 className="text-3xl font-bold mt-1">{result.diagnosis}</h2>
                  <p className="text-muted-foreground mt-2 max-w-3xl">
                    {result.message}
                  </p>

                  <div className="flex flex-wrap items-center justify-center xl:justify-start gap-4 mt-5">
                    <div className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-primary" />
                      <span className="text-2xl font-bold">{result.score}</span>
                      <span className="text-muted-foreground">/100</span>
                    </div>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Clock className="w-4 h-4" />
                      <span className="text-sm">Just now</span>
                    </div>
                    {result.risk_level && (
                      <div className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                        {result.risk_level}
                      </div>
                    )}
                    {result.quality_info?.audio_quality_label && (
                      <div className="px-3 py-1 rounded-full bg-secondary text-secondary-foreground text-sm font-medium">
                        Audio: {result.quality_info.audio_quality_label}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
            {[
              { label: "Pitch Quality", value: result.metrics.pitch, icon: Waves },
              {
                label: "Voice Clarity",
                value: result.metrics.clarity,
                icon: Volume2,
              },
              {
                label: "Strain Level",
                value: result.metrics.strain,
                icon: AlertTriangle,
                inverse: true,
              },
              {
                label: "Stability",
                value: result.metrics.stability,
                icon: Activity,
              },
            ].map((metric, index) => (
              <Card key={index} className="bg-card border-border">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 text-muted-foreground mb-2">
                    <metric.icon className="w-4 h-4" />
                    <span className="text-xs">{metric.label}</span>
                  </div>
                  <div className="flex items-end gap-1">
                    <span className="text-2xl font-bold">{metric.value}</span>
                    <span className="text-sm text-muted-foreground mb-1">
                      /100
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        metric.inverse
                          ? metric.value > 30
                            ? "bg-yellow-500"
                            : "bg-green-500"
                          : metric.value >= 80
                          ? "bg-green-500"
                          : "bg-yellow-500"
                      }`}
                      style={{
                        width: `${Math.max(0, Math.min(100, metric.value))}%`,
                      }}
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Recommendations</CardTitle>
                <CardDescription>
                  Based on the current screening result
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {result.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-xs font-medium text-primary">
                          {index + 1}
                        </span>
                      </div>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Detailed Acoustic Metrics</CardTitle>
                <CardDescription>
                  Measured indicators from the uploaded voice sample
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span>Pitch Mean</span>
                  <span>{result.raw_metrics?.pitch_mean_hz ?? "-"} Hz</span>
                </div>
                <div className="flex justify-between">
                  <span>Pitch Std Dev</span>
                  <span>{result.raw_metrics?.pitch_std_dev ?? "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Jitter</span>
                  <span>{result.raw_metrics?.jitter ?? "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Shimmer</span>
                  <span>{result.raw_metrics?.shimmer ?? "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span>HNR</span>
                  <span>{result.raw_metrics?.hnr ?? "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span>AVHI</span>
                  <span>{result.raw_metrics?.avhi ?? "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration</span>
                  <span>{result.quality_info?.duration_sec ?? "-"} sec</span>
                </div>
                <div className="flex justify-between">
                  <span>Audio Quality</span>
                  <span>{result.quality_info?.audio_quality_label ?? "-"}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {!!result.insights?.length && (
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Screening Insights</CardTitle>
                <CardDescription>
                  Acoustic interpretation based on extracted features
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {result.insights.map((insight, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="mt-1 text-primary">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          <Card className="bg-card border-border">
            <CardContent className="p-6 text-sm text-muted-foreground">
              This is an AI-based screening output and not a confirmed medical
              diagnosis. For persistent hoarseness, pain, vocal strain, or
              fatigue, please consult an ENT specialist or speech expert.
            </CardContent>
          </Card>

          <div className="flex flex-col sm:flex-row gap-4">
            <Button
              onClick={resetAnalysis}
              className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              New Analysis
            </Button>

            <Button
              variant="outline"
              onClick={() => result && generatePDF(result)}
              className="flex-1 border-border bg-transparent hover:bg-primary/10"
            >
              Download Report
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

import jsPDF from "jspdf";

const generatePDF = (result: AnalysisResult) => {
  const doc = new jsPDF();

  doc.setFontSize(18);
  doc.text("Vocare Voice Health Report", 20, 20);

  doc.setFontSize(12);

  doc.text(`Diagnosis: ${result.diagnosis}`, 20, 40);
  doc.text(`Confidence: ${result.score}`, 20, 50);
  doc.text(`Health Score: ${result.health_score}`, 20, 60);
  doc.text(`Risk Level: ${result.risk_level}`, 20, 70);

  doc.text("Metrics:", 20, 90);
  doc.text(`Pitch: ${result.metrics.pitch}`, 20, 100);
  doc.text(`Clarity: ${result.metrics.clarity}`, 20, 110);
  doc.text(`Strain: ${result.metrics.strain}`, 20, 120);
  doc.text(`Stability: ${result.metrics.stability}`, 20, 130);

  doc.text("Insights:", 20, 150);
  result.insights?.forEach((ins, i) => {
    doc.text(`- ${ins}`, 20, 160 + i * 10);
  });

  doc.text("Recommendations:", 20, 200);
  result.recommendations.forEach((rec, i) => {
    doc.text(`- ${rec}`, 20, 210 + i * 10);
  });

  doc.save("vocare_report.pdf");
};

const audioBufferToWav = (buffer: AudioBuffer) => {
  const numOfChan = buffer.numberOfChannels;
  const length = buffer.length * numOfChan * 2 + 44;
  const bufferArray = new ArrayBuffer(length);
  const view = new DataView(bufferArray);

  let offset = 0;

  const writeString = (str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset++, str.charCodeAt(i));
    }
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + buffer.length * 2, true);
  offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint16(offset, numOfChan, true);
  offset += 2;
  view.setUint32(offset, buffer.sampleRate, true);
  offset += 4;
  view.setUint32(offset, buffer.sampleRate * 2 * numOfChan, true);
  offset += 4;
  view.setUint16(offset, numOfChan * 2, true);
  offset += 2;
  view.setUint16(offset, 16, true);
  offset += 2;
  writeString("data");
  view.setUint32(offset, buffer.length * numOfChan * 2, true);
  offset += 4;

  const channels = [];
  for (let i = 0; i < numOfChan; i++) {
    channels.push(buffer.getChannelData(i));
  }

  let sampleIndex = 0;
  while (sampleIndex < buffer.length) {
    for (let channel = 0; channel < numOfChan; channel++) {
      let sample = Math.max(-1, Math.min(1, channels[channel][sampleIndex]));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, sample, true);
      offset += 2;
    }
    sampleIndex++;
  }

  return new Blob([view], { type: "audio/wav" });
};