"use client";

import type { FileUIPart, SourceDocumentUIPart } from "ai";
import type { ComponentProps, HTMLAttributes, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";
import {
  FileArchiveIcon,
  FileCodeIcon,
  FileSpreadsheetIcon,
  FileTextIcon,
  GlobeIcon,
  ImageIcon,
  Music2Icon,
  PaperclipIcon,
  PresentationIcon,
  VideoIcon,
  XIcon,
} from "lucide-react";
import { createContext, useCallback, useContext, useMemo } from "react";

// ============================================================================
// Types
// ============================================================================

export type AttachmentData =
  | (FileUIPart & { id: string })
  | (SourceDocumentUIPart & { id: string });

export type AttachmentMediaCategory =
  | "image"
  | "video"
  | "audio"
  | "document"
  | "source"
  | "unknown";

export type AttachmentVariant = "grid" | "inline" | "list";

const mediaCategoryIcons: Record<AttachmentMediaCategory, typeof ImageIcon> = {
  audio: Music2Icon,
  document: FileTextIcon,
  image: ImageIcon,
  source: GlobeIcon,
  unknown: PaperclipIcon,
  video: VideoIcon,
};

// ============================================================================
// Utility Functions
// ============================================================================

export const getMediaCategory = (
  data: AttachmentData
): AttachmentMediaCategory => {
  if (data.type === "source-document") {
    return "source";
  }

  const mediaType = data.mediaType ?? "";

  if (mediaType.startsWith("image/")) {
    return "image";
  }
  if (mediaType.startsWith("video/")) {
    return "video";
  }
  if (mediaType.startsWith("audio/")) {
    return "audio";
  }
  if (mediaType.startsWith("application/") || mediaType.startsWith("text/")) {
    return "document";
  }

  return "unknown";
};

export const getAttachmentLabel = (data: AttachmentData): string => {
  if (data.type === "source-document") {
    return data.title || data.filename || "Source";
  }

  const category = getMediaCategory(data);
  return data.filename || (category === "image" ? "Image" : "Attachment");
};

const renderAttachmentImage = (
  url: string,
  filename: string | undefined,
  isGrid: boolean
) =>
  isGrid ? (
    <img
      alt={filename || "Image"}
      className="size-full object-cover transition-[filter,transform] group-hover/attachment:brightness-75"
      height={96}
      src={url}
      width={96}
    />
  ) : (
    <img
      alt={filename || "Image"}
      className="size-full rounded object-cover transition-[filter,transform] group-hover/attachment:brightness-90"
      height={20}
      src={url}
      width={20}
    />
  );

type DocumentAttachmentThemeId =
  | "pdf"
  | "word"
  | "sheet"
  | "slides"
  | "archive"
  | "code"
  | "text";

type DocumentAttachmentConfig = {
  label: string;
  themeId: DocumentAttachmentThemeId;
};

const DOCUMENT_THEME_CONFIG: Record<
  DocumentAttachmentThemeId,
  {
    accentClassName: string;
    badgeClassName: string;
    icon: typeof FileTextIcon;
  }
> = {
  archive: {
    accentClassName:
      "border-orange-500/25 bg-orange-500/10 text-orange-700 dark:text-orange-300",
    badgeClassName: "bg-orange-500/15 text-orange-700 dark:text-orange-300",
    icon: FileArchiveIcon,
  },
  code: {
    accentClassName:
      "border-cyan-500/25 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
    badgeClassName: "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300",
    icon: FileCodeIcon,
  },
  pdf: {
    accentClassName:
      "border-rose-500/25 bg-rose-500/10 text-rose-700 dark:text-rose-300",
    badgeClassName: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
    icon: FileTextIcon,
  },
  sheet: {
    accentClassName:
      "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    badgeClassName:
      "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
    icon: FileSpreadsheetIcon,
  },
  slides: {
    accentClassName:
      "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    badgeClassName: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
    icon: PresentationIcon,
  },
  text: {
    accentClassName:
      "border-slate-500/25 bg-slate-500/10 text-slate-700 dark:text-slate-300",
    badgeClassName: "bg-slate-500/15 text-slate-700 dark:text-slate-300",
    icon: FileTextIcon,
  },
  word: {
    accentClassName:
      "border-sky-500/25 bg-sky-500/10 text-sky-700 dark:text-sky-300",
    badgeClassName: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
    icon: FileTextIcon,
  },
};

const getFilenameExtension = (filename?: string): string => {
  const normalized = filename?.trim().toLowerCase() ?? "";
  const extension = normalized.split(".").pop();

  if (!extension || extension === normalized) {
    return "";
  }

  return extension;
};

const getDocumentAttachmentFormat = (
  data: AttachmentData
): DocumentAttachmentConfig | null => {
  if (data.type !== "file") {
    return null;
  }

  const mediaType = data.mediaType?.toLowerCase() ?? "";
  const extension = getFilenameExtension(data.filename);

  if (mediaType === "application/pdf" || extension === "pdf") {
    return { label: "PDF", themeId: "pdf" };
  }

  if (
    mediaType === "application/msword" ||
    mediaType === "application/rtf" ||
    mediaType === "application/vnd.oasis.opendocument.text" ||
    extension === "doc" ||
    extension === "odt" ||
    extension === "rtf"
  ) {
    return { label: extension ? extension.toUpperCase() : "DOC", themeId: "word" };
  }

  if (
    mediaType ===
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    extension === "docx"
  ) {
    return { label: "DOCX", themeId: "word" };
  }

  if (
    mediaType === "application/vnd.ms-excel" ||
    mediaType ===
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ||
    mediaType === "text/csv" ||
    mediaType === "text/tab-separated-values" ||
    mediaType === "application/vnd.oasis.opendocument.spreadsheet" ||
    ["csv", "ods", "numbers", "tsv", "xls", "xlsx"].includes(extension)
  ) {
    return {
      label: extension ? extension.toUpperCase() : "XLS",
      themeId: "sheet",
    };
  }

  if (
    mediaType === "application/vnd.ms-powerpoint" ||
    mediaType ===
      "application/vnd.openxmlformats-officedocument.presentationml.presentation" ||
    mediaType === "application/vnd.oasis.opendocument.presentation" ||
    ["key", "odp", "ppt", "pptx"].includes(extension)
  ) {
    return {
      label: extension ? extension.toUpperCase() : "PPT",
      themeId: "slides",
    };
  }

  if (
    [
      "application/gzip",
      "application/vnd.rar",
      "application/x-7z-compressed",
      "application/x-bzip2",
      "application/x-rar-compressed",
      "application/x-tar",
      "application/x-zip-compressed",
      "application/zip",
    ].includes(mediaType) ||
    ["7z", "bz2", "gz", "rar", "tar", "tgz", "xz", "zip"].includes(extension)
  ) {
    return {
      label: extension ? extension.toUpperCase() : "ZIP",
      themeId: "archive",
    };
  }

  if (
    [
      "application/json",
      "application/ld+json",
      "application/sql",
      "application/toml",
      "application/x-httpd-php",
      "application/xml",
      "text/css",
      "text/html",
      "text/javascript",
      "text/jsx",
      "text/markdown",
      "text/typescript",
      "text/x-python",
      "text/xml",
      "text/yaml",
    ].includes(mediaType) ||
    [
      "bash",
      "c",
      "cc",
      "conf",
      "cpp",
      "cs",
      "css",
      "go",
      "h",
      "hpp",
      "html",
      "ini",
      "java",
      "js",
      "json",
      "jsx",
      "kt",
      "kts",
      "less",
      "md",
      "php",
      "py",
      "rb",
      "rs",
      "sass",
      "scss",
      "sh",
      "sql",
      "swift",
      "toml",
      "ts",
      "tsx",
      "xml",
      "yaml",
      "yml",
      "zsh",
    ].includes(extension)
  ) {
    return {
      label: extension ? extension.toUpperCase() : "CODE",
      themeId: "code",
    };
  }

  if (
    mediaType.startsWith("text/") ||
    ["log", "txt"].includes(extension)
  ) {
    return {
      label: extension ? extension.toUpperCase() : "TXT",
      themeId: "text",
    };
  }

  return null;
};

const renderDocumentAttachmentIcon = (
  format: DocumentAttachmentConfig,
  variant: AttachmentVariant
) => {
  const theme = DOCUMENT_THEME_CONFIG[format.themeId];
  const Icon = theme.icon;

  if (variant === "inline") {
    return (
      <div
        className={cn(
          "flex size-full items-center justify-center rounded border transition-[filter,colors] group-hover/attachment:brightness-90",
          theme.accentClassName
        )}
      >
        <Icon className="size-3.5" />
      </div>
    );
  }

  if (variant === "list") {
    return (
      <div
        className={cn(
          "flex size-full flex-col items-center justify-center gap-1 rounded border transition-[filter,colors] group-hover/attachment:brightness-90",
          theme.accentClassName
        )}
      >
        <Icon className="size-5" />
        <span
          className={cn(
            "rounded-full px-1.5 py-0.5 text-[10px] font-semibold tracking-[0.08em]",
            theme.badgeClassName
          )}
        >
          {format.label}
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex size-full flex-col items-center justify-center gap-2 rounded-lg border transition-[filter,colors] group-hover/attachment:brightness-90",
        theme.accentClassName
      )}
    >
      <Icon className="size-7" />
      <span
        className={cn(
          "rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-[0.08em]",
          theme.badgeClassName
        )}
      >
        {format.label}
      </span>
    </div>
  );
};

// ============================================================================
// Contexts
// ============================================================================

interface AttachmentsContextValue {
  variant: AttachmentVariant;
}

const AttachmentsContext = createContext<AttachmentsContextValue | null>(null);

interface AttachmentContextValue {
  data: AttachmentData;
  mediaCategory: AttachmentMediaCategory;
  onRemove?: () => void;
  variant: AttachmentVariant;
}

const AttachmentContext = createContext<AttachmentContextValue | null>(null);

// ============================================================================
// Hooks
// ============================================================================

export const useAttachmentsContext = () =>
  useContext(AttachmentsContext) ?? { variant: "grid" as const };

export const useAttachmentContext = () => {
  const ctx = useContext(AttachmentContext);
  if (!ctx) {
    throw new Error("Attachment components must be used within <Attachment>");
  }
  return ctx;
};

// ============================================================================
// Attachments - Container
// ============================================================================

export type AttachmentsProps = HTMLAttributes<HTMLDivElement> & {
  variant?: AttachmentVariant;
};

export const Attachments = ({
  variant = "grid",
  className,
  children,
  ...props
}: AttachmentsProps) => {
  const contextValue = useMemo(() => ({ variant }), [variant]);

  return (
    <AttachmentsContext.Provider value={contextValue}>
      <div
        className={cn(
          "flex items-start",
          variant === "list" ? "flex-col gap-2" : "flex-wrap gap-2",
          variant === "grid" && "ml-auto w-fit",
          className
        )}
        {...props}
      >
        {children}
      </div>
    </AttachmentsContext.Provider>
  );
};

// ============================================================================
// Attachment - Item
// ============================================================================

export type AttachmentProps = HTMLAttributes<HTMLDivElement> & {
  data: AttachmentData;
  onRemove?: () => void;
};

export const Attachment = ({
  data,
  onRemove,
  className,
  children,
  ...props
}: AttachmentProps) => {
  const { variant } = useAttachmentsContext();
  const mediaCategory = getMediaCategory(data);

  const contextValue = useMemo<AttachmentContextValue>(
    () => ({ data, mediaCategory, onRemove, variant }),
    [data, mediaCategory, onRemove, variant]
  );

  return (
    <AttachmentContext.Provider value={contextValue}>
      <div
        className={cn(
          "group/attachment relative",
          variant === "grid" && [
            "size-24 overflow-hidden rounded-lg transition-colors",
            "hover:bg-muted/80",
          ],
          variant === "inline" && [
            "flex h-8 cursor-pointer select-none items-center gap-1.5",
            "rounded-md border border-border px-1.5",
            "font-medium text-sm transition-all",
            "hover:bg-accent/90 hover:text-accent-foreground dark:hover:bg-accent/70",
          ],
          variant === "list" && [
            "flex w-full items-center gap-3 rounded-lg border p-3 transition-colors",
            "hover:bg-accent/80",
          ],
          className
        )}
        {...props}
      >
        {children}
      </div>
    </AttachmentContext.Provider>
  );
};

// ============================================================================
// AttachmentPreview - Media preview
// ============================================================================

export type AttachmentPreviewProps = HTMLAttributes<HTMLDivElement> & {
  fallbackIcon?: ReactNode;
};

export const AttachmentPreview = ({
  fallbackIcon,
  className,
  ...props
}: AttachmentPreviewProps) => {
  const { data, mediaCategory, variant } = useAttachmentContext();
  const documentFormat = getDocumentAttachmentFormat(data);

  const iconSize = variant === "inline" ? "size-3" : "size-4";

  const renderIcon = (Icon: typeof ImageIcon) => (
    <Icon className={cn(iconSize, "text-muted-foreground")} />
  );

  const renderContent = () => {
    if (mediaCategory === "image" && data.type === "file" && data.url) {
      return renderAttachmentImage(data.url, data.filename, variant === "grid");
    }

    if (mediaCategory === "video" && data.type === "file" && data.url) {
      return (
        <video
          className="size-full object-cover transition-[filter,transform] group-hover/attachment:brightness-75"
          muted
          src={data.url}
        />
      );
    }

    if (documentFormat) {
      return renderDocumentAttachmentIcon(documentFormat, variant);
    }

    const Icon = mediaCategoryIcons[mediaCategory];
    return fallbackIcon ?? renderIcon(Icon);
  };

  return (
    <div
      className={cn(
        "flex shrink-0 items-center justify-center overflow-hidden transition-[filter,colors]",
        variant === "grid" && "size-full bg-muted group-hover/attachment:bg-muted/80",
        variant === "inline" && "size-5 rounded bg-background group-hover/attachment:bg-background/80",
        variant === "list" && "size-12 rounded bg-muted group-hover/attachment:bg-muted/80",
        className
      )}
      {...props}
    >
      {renderContent()}
    </div>
  );
};

// ============================================================================
// AttachmentInfo - Name and type display
// ============================================================================

export type AttachmentInfoProps = HTMLAttributes<HTMLDivElement> & {
  showMediaType?: boolean;
};

export const AttachmentInfo = ({
  showMediaType = false,
  className,
  ...props
}: AttachmentInfoProps) => {
  const { data, variant } = useAttachmentContext();
  const label = getAttachmentLabel(data);

  if (variant === "grid") {
    return null;
  }

  return (
    <div className={cn("min-w-0 flex-1", className)} {...props}>
      <span className="block truncate">{label}</span>
      {showMediaType && data.mediaType && (
        <span className="block truncate text-muted-foreground text-xs">
          {data.mediaType}
        </span>
      )}
    </div>
  );
};

// ============================================================================
// AttachmentRemove - Remove button
// ============================================================================

export type AttachmentRemoveProps = ComponentProps<typeof Button> & {
  label?: string;
};

export const AttachmentRemove = ({
  label = "Remove",
  className,
  children,
  ...props
}: AttachmentRemoveProps) => {
  const { onRemove, variant } = useAttachmentContext();

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onRemove?.();
    },
    [onRemove]
  );

  if (!onRemove) {
    return null;
  }

  return (
    <Button
      aria-label={label}
      className={cn(
        variant === "grid" && [
          "absolute top-2 right-2 size-6 rounded-full p-0",
          "bg-background/80 backdrop-blur-sm",
          "opacity-0 transition-opacity group-hover/attachment:opacity-100",
          "hover:bg-background",
          "[&>svg]:size-3",
        ],
        variant === "inline" && [
          "size-5 rounded p-0",
          "opacity-0 transition-opacity group-hover/attachment:opacity-100",
          "[&>svg]:size-2.5",
        ],
        variant === "list" && ["size-8 shrink-0 rounded p-0", "[&>svg]:size-4"],
        className
      )}
      onClick={handleClick}
      type="button"
      variant="ghost"
      {...props}
    >
      {children ?? <XIcon />}
      <span className="sr-only">{label}</span>
    </Button>
  );
};

// ============================================================================
// AttachmentHoverCard - Hover preview
// ============================================================================

export type AttachmentHoverCardProps = ComponentProps<typeof HoverCard>;

export const AttachmentHoverCard = (
  props: AttachmentHoverCardProps
) => (
  <HoverCard {...props} />
);

export type AttachmentHoverCardTriggerProps = ComponentProps<
  typeof HoverCardTrigger
>;

export const AttachmentHoverCardTrigger = (
  props: AttachmentHoverCardTriggerProps
) => <HoverCardTrigger {...props} />;

export type AttachmentHoverCardContentProps = ComponentProps<
  typeof HoverCardContent
>;

export const AttachmentHoverCardContent = ({
  align = "start",
  className,
  ...props
}: AttachmentHoverCardContentProps) => (
  <HoverCardContent
    align={align}
    className={cn("w-auto p-2", className)}
    {...props}
  />
);

// ============================================================================
// AttachmentEmpty - Empty state
// ============================================================================

export type AttachmentEmptyProps = HTMLAttributes<HTMLDivElement>;

export const AttachmentEmpty = ({
  className,
  children,
  ...props
}: AttachmentEmptyProps) => (
  <div
    className={cn(
      "flex items-center justify-center p-4 text-muted-foreground text-sm",
      className
    )}
    {...props}
  >
    {children ?? "No attachments"}
  </div>
);
