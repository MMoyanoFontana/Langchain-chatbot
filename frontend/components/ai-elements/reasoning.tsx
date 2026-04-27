"use client";

import type { ComponentProps } from "react";

import { useControllableState } from "@radix-ui/react-use-controllable-state";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { BrainIcon, ChevronDownIcon } from "lucide-react";
import {
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Streamdown } from "streamdown";

import { Shimmer } from "@/components/ai-elements/shimmer";

interface ReasoningContextValue {
  isStreaming: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  duration: number;
}

const ReasoningContext = createContext<ReasoningContextValue | null>(null);

const useReasoning = () => {
  const ctx = useContext(ReasoningContext);
  if (!ctx) {
    throw new Error("Reasoning components must be used within <Reasoning>");
  }
  return ctx;
};

const AUTO_CLOSE_DELAY_MS = 1000;

export type ReasoningProps = ComponentProps<typeof Collapsible> & {
  isStreaming?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  duration?: number;
};

export const Reasoning = memo(
  ({
    className,
    isStreaming = false,
    open,
    defaultOpen = true,
    onOpenChange,
    duration: durationProp,
    children,
    ...props
  }: ReasoningProps) => {
    const [isOpen, setIsOpen] = useControllableState({
      prop: open,
      defaultProp: defaultOpen,
      onChange: onOpenChange,
    });

    // Track streaming duration from the first render where isStreaming is true
    // to the one where it flips false. durationProp overrides if the caller
    // wants to control it.
    const [internalDuration, setInternalDuration] = useState(0);
    const startedAtRef = useRef<number | null>(null);
    const hasAutoClosedRef = useRef(false);
    // Tracks whether isStreaming was ever true so auto-close fires even when
    // thinking completes in under 500ms (internalDuration rounds to 0 on Groq).
    const hasStartedStreamingRef = useRef(false);

    useEffect(() => {
      if (isStreaming) {
        hasStartedStreamingRef.current = true;
        if (startedAtRef.current === null) {
          startedAtRef.current = performance.now();
        }
        const interval = window.setInterval(() => {
          if (startedAtRef.current !== null) {
            setInternalDuration(
              Math.round((performance.now() - startedAtRef.current) / 1000)
            );
          }
        }, 250);
        return () => window.clearInterval(interval);
      }

      if (startedAtRef.current !== null) {
        setInternalDuration(
          Math.round((performance.now() - startedAtRef.current) / 1000)
        );
        startedAtRef.current = null;
      }
    }, [isStreaming]);

    // Auto-collapse shortly after streaming finishes, so the user sees the
    // answer rather than a long reasoning panel. Only runs once per lifecycle.
    useEffect(() => {
      if (isStreaming || hasAutoClosedRef.current) return;
      if (!hasStartedStreamingRef.current) return;
      hasAutoClosedRef.current = true;
      const timer = window.setTimeout(() => setIsOpen(false), AUTO_CLOSE_DELAY_MS);
      return () => window.clearTimeout(timer);
    }, [isStreaming, setIsOpen]);

    const duration = durationProp ?? internalDuration;

    const contextValue = useMemo<ReasoningContextValue>(
      () => ({
        isStreaming,
        isOpen: Boolean(isOpen),
        setIsOpen,
        duration,
      }),
      [isStreaming, isOpen, setIsOpen, duration]
    );

    return (
      <ReasoningContext.Provider value={contextValue}>
        <Collapsible
          className={cn("not-prose mb-3", className)}
          open={isOpen}
          onOpenChange={setIsOpen}
          {...props}
        >
          {children}
        </Collapsible>
      </ReasoningContext.Provider>
    );
  }
);

Reasoning.displayName = "Reasoning";

export type ReasoningTriggerProps = ComponentProps<typeof CollapsibleTrigger> & {
  title?: string;
};

export const ReasoningTrigger = memo(
  ({ className, title, children, ...props }: ReasoningTriggerProps) => {
    const { isStreaming, isOpen, duration } = useReasoning();
    const label = useMemo(() => {
      if (title) return title;
      if (isStreaming) return "Thinking…";
      if (duration > 0) return `Thought for ${duration}s`;
      return "Reasoning";
    }, [title, isStreaming, duration]);

    return (
      <CollapsibleTrigger
        className={cn(
          "text-muted-foreground hover:text-foreground flex items-center gap-2 text-xs transition-colors",
          className
        )}
        {...props}
      >
        <BrainIcon className="size-3.5 shrink-0" />
        {children ??
          (isStreaming ? (
            <Shimmer duration={1.2} className="text-xs">
              {label}
            </Shimmer>
          ) : (
            <span>{label}</span>
          ))}
        <ChevronDownIcon
          className={cn(
            "size-3.5 shrink-0 transition-transform",
            isOpen ? "rotate-180" : undefined
          )}
        />
      </CollapsibleTrigger>
    );
  }
);

ReasoningTrigger.displayName = "ReasoningTrigger";

export type ReasoningContentProps = ComponentProps<"div"> & {
  children: string;
};

export const ReasoningContent = memo(
  ({ className, children, ...props }: ReasoningContentProps) => (
    <CollapsibleContent
      className={cn(
        "text-muted-foreground mt-2 border-l-2 pl-3 text-sm",
        "data-[state=open]:animate-in data-[state=closed]:animate-out",
        "data-[state=open]:fade-in-0 data-[state=closed]:fade-out-0",
        className
      )}
      {...props}
    >
      <Streamdown className="[&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
        {children}
      </Streamdown>
    </CollapsibleContent>
  )
);

ReasoningContent.displayName = "ReasoningContent";
