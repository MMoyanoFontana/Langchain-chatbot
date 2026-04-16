"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";

const MAX_SYSTEM_PROMPT_LENGTH = 4000;

type SystemPromptDialogProps = {
  open: boolean;
  value: string;
  requestError?: string | null;
  isSubmitting: boolean;
  onValueChange: (value: string) => void;
  onSubmit: () => void;
  onClose: () => void;
};

export function SystemPromptDialog({
  open,
  value,
  requestError,
  isSubmitting,
  onValueChange,
  onSubmit,
  onClose,
}: SystemPromptDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !isSubmitting) {
          onClose();
        }
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogTitle>System prompt</DialogTitle>
        <DialogDescription>
          Set instructions the assistant will follow for this chat. Leave blank to remove.
        </DialogDescription>
        <Textarea
          value={value}
          onChange={(event) => onValueChange(event.target.value)}
          disabled={isSubmitting}
          maxLength={MAX_SYSTEM_PROMPT_LENGTH}
          placeholder="You are a helpful assistant specialized in..."
          className="min-h-32 resize-y font-mono text-sm"
          autoFocus
        />
        <p className="text-xs text-muted-foreground text-right">
          {value.length} / {MAX_SYSTEM_PROMPT_LENGTH}
        </p>
        {requestError ? (
          <p className="text-sm text-destructive">{requestError}</p>
        ) : null}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button type="button" onClick={onSubmit} disabled={isSubmitting}>
            {isSubmitting ? "Saving..." : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
