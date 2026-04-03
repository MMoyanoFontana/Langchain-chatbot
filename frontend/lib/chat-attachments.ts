import type { FileUIPart } from "ai";

export type ChatAttachment = FileUIPart;

export type BackendChatAttachment = {
  type?: string | null;
  filename?: string | null;
  media_type?: string | null;
  mediaType?: string | null;
  url?: string | null;
};

export type ChatAttachmentRequest = {
  type: "file";
  filename?: string | null;
  media_type: string;
  url: string;
};

export const fileToDataUrl = async (file: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === "string" && reader.result) {
        resolve(reader.result);
        return;
      }
      reject(new Error("Attachment could not be converted to a data URL."));
    };
    reader.onerror = () =>
      reject(reader.error ?? new Error("Attachment could not be converted to a data URL."));
    reader.readAsDataURL(file);
  });

export const toBackendChatAttachment = (
  attachment: FileUIPart
): ChatAttachmentRequest => ({
  type: "file",
  filename: attachment.filename ?? null,
  media_type: attachment.mediaType,
  url: attachment.url,
});

export const toChatAttachment = (
  attachment: BackendChatAttachment
): ChatAttachment | null => {
  const mediaType =
    attachment.media_type?.trim() || attachment.mediaType?.trim() || "";
  const url = attachment.url?.trim() || "";

  if ((attachment.type && attachment.type !== "file") || !mediaType || !url) {
    return null;
  }

  return {
    type: "file",
    filename: attachment.filename?.trim() || undefined,
    mediaType,
    url,
  };
};
