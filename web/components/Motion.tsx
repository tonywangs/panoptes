"use client";

import {
  HTMLMotionProps,
  motion,
  useInView,
  useMotionValue,
  useScroll,
  useSpring,
  useTransform,
} from "motion/react";
import { ReactNode, useEffect, useRef, useState } from "react";

/**
 * Drop-in wrappers that animate content as the user scrolls.
 * All client components — server components import them as-is.
 */

const EASE = [0.22, 1, 0.36, 1] as const;

export function FadeIn({
  children,
  delay = 0,
  y = 18,
  className,
  as = "div",
  ...rest
}: {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
  as?: "div" | "section" | "header" | "article";
} & HTMLMotionProps<"div">) {
  const Tag = motion[as] as typeof motion.div;
  return (
    <Tag
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.55, ease: EASE, delay }}
      className={className}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export function Stagger({
  children,
  className,
  stagger = 0.06,
}: {
  children: ReactNode;
  className?: string;
  stagger?: number;
}) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-60px" }}
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: stagger } },
      }}
    >
      {children}
    </motion.div>
  );
}

export function StaggerChild({
  children,
  className,
  y = 14,
}: {
  children: ReactNode;
  className?: string;
  y?: number;
}) {
  return (
    <motion.div
      className={className}
      variants={{
        hidden: { opacity: 0, y },
        visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE } },
      }}
    >
      {children}
    </motion.div>
  );
}

/** Animated number that counts up to `to` when scrolled into view.
 * `format` is a string discriminator (not a function) so this component can
 * be used directly from server components without serializing a callback
 * across the RSC boundary. */
type CountFormat = "int" | "intCommas" | "usd" | "percent" | "decimal2" | "decimal3";

function formatValue(v: number, kind: CountFormat): string {
  switch (kind) {
    case "int":
      return Math.round(v).toString();
    case "intCommas":
      return Math.round(v).toLocaleString();
    case "usd":
      if (v < 0.01) return `$${v.toFixed(4)}`;
      if (v < 1) return `$${v.toFixed(3)}`;
      return `$${v.toFixed(2)}`;
    case "percent":
      return `${Math.round(v)}%`;
    case "decimal2":
      return v.toFixed(2);
    case "decimal3":
      return v.toFixed(3);
  }
}

export function CountUp({
  to,
  format = "int",
  duration = 1.4,
  className,
}: {
  to: number;
  format?: CountFormat;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-30%" });
  const [display, setDisplay] = useState(formatValue(0, format));

  useEffect(() => {
    if (!inView) return;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / (duration * 1000));
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(formatValue(to * eased, format));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, to, duration, format]);

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  );
}

/** Thin progress bar at the very top of the page, tied to vertical scroll. */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const width = useSpring(scrollYProgress, { stiffness: 220, damping: 28, mass: 0.4 });
  const widthPct = useTransform(width, (v) => `${v * 100}%`);
  return (
    <motion.div
      style={{ width: widthPct }}
      className="fixed top-0 left-0 h-[2px] bg-emerald-500 z-50 origin-left"
    />
  );
}

/**
 * Tilts and lifts a card slightly on hover, with a magnetic cursor-follow
 * tilt that resets on leave. Use for hero CTAs / featured cards.
 */
export function TiltCard({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const rx = useMotionValue(0);
  const ry = useMotionValue(0);
  const rotateX = useSpring(rx, { stiffness: 200, damping: 18 });
  const rotateY = useSpring(ry, { stiffness: 200, damping: 18 });

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    rx.set(py * -6);
    ry.set(px * 6);
  };
  const onLeave = () => {
    rx.set(0);
    ry.set(0);
  };

  return (
    <motion.div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{ rotateX, rotateY, transformPerspective: 900 }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/** Subtle parallax on background decorations as the user scrolls. */
export function Parallax({
  children,
  speed = 0.25,
  className,
}: {
  children: ReactNode;
  speed?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], [`${speed * -60}px`, `${speed * 60}px`]);
  return (
    <motion.div ref={ref} style={{ y }} className={className}>
      {children}
    </motion.div>
  );
}
