"use client";

// Small animation helpers shared by the pages: 3D tilt card, count-up number, fade-up reveal.
import { animate, motion, useMotionTemplate, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export function TiltCard({ children, className = "", max = 7 }: { children: React.ReactNode; className?: string; max?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const x = useMotionValue(0.5);
  const y = useMotionValue(0.5);
  const rx = useSpring(useTransform(y, [0, 1], [max, -max]), { stiffness: 160, damping: 18 });
  const ry = useSpring(useTransform(x, [0, 1], [-max, max]), { stiffness: 160, damping: 18 });
  const glowX = useTransform(x, (v) => `${v * 100}%`);
  const glowY = useTransform(y, (v) => `${v * 100}%`);
  const glow = useMotionTemplate`radial-gradient(420px circle at ${glowX} ${glowY}, rgba(61,220,151,0.10), transparent 60%)`;
  return (
    <div style={{ perspective: 1200 }}>
      <motion.div
        ref={ref}
        onMouseMove={(e) => {
          const r = ref.current!.getBoundingClientRect();
          x.set((e.clientX - r.left) / r.width);
          y.set((e.clientY - r.top) / r.height);
        }}
        onMouseLeave={() => {
          x.set(0.5);
          y.set(0.5);
        }}
        style={{ rotateX: rx, rotateY: ry, transformStyle: "preserve-3d" }}
        className={`relative ${className}`}
      >
        <motion.div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-[18px] opacity-60"
          style={{ background: glow }}
        />
        <div style={{ transform: "translateZ(30px)" }}>{children}</div>
      </motion.div>
    </div>
  );
}

export function CountUp({ value, decimals = 0, suffix = "", className = "" }: { value: number; decimals?: number; suffix?: string; className?: string }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const c = animate(0, value, { duration: 1.1, ease: [0.16, 1, 0.3, 1], onUpdate: setShown });
    return () => c.stop();
  }, [value]);
  return (
    <span className={className}>
      {shown.toFixed(decimals)}
      {suffix}
    </span>
  );
}

export function Reveal({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24, filter: "blur(6px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export function Bar({ value, className = "", delay = 0 }: { value: number; className?: string; delay?: number }) {
  return (
    <motion.span
      className={`block h-full rounded-full ${className}`}
      initial={{ width: 0 }}
      animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      transition={{ duration: 0.9, delay, ease: [0.16, 1, 0.3, 1] }}
    />
  );
}
