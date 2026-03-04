"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog";

type DeleteThreadDialogProps = {
  open: boolean;
  threadTitle?: string | null;
  error?: string | null;
  isSubmitting: boolean;
  canSubmit: boolean;
  onSubmit: () => void;
  onClose: () => void;
};

export function DeleteThreadDialog({
  open,
  threadTitle,
  error,
  isSubmitting,
  canSubmit,
  onSubmit,
  onClose,
}: DeleteThreadDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !isSubmitting) {
          onClose();
        }
      }}
    >
      <DialogContent
        className="sm:max-w-md"
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            if (canSubmit) {
              onSubmit();
            }
          }
          if (event.key === "Escape") {
            event.preventDefault();
            onClose();
          }
        }}
      >
        <DialogTitle>Delete chat?</DialogTitle>
        <DialogDescription>
          This will permanently remove
          {threadTitle ? ` "${threadTitle}"` : " this chat"}.
        </DialogDescription>
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={onSubmit}
            disabled={!canSubmit}
          >
            {isSubmitting ? "Deleting..." : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
