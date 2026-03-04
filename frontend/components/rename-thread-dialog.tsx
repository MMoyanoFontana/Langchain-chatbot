"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

type RenameThreadDialogProps = {
  open: boolean;
  value: string;
  validationError?: string | null;
  requestError?: string | null;
  isSubmitting: boolean;
  canSubmit: boolean;
  maxLength: number;
  onValueChange: (value: string) => void;
  onSubmit: () => void;
  onClose: () => void;
};

export function RenameThreadDialog({
  open,
  value,
  validationError,
  requestError,
  isSubmitting,
  canSubmit,
  maxLength,
  onValueChange,
  onSubmit,
  onClose,
}: RenameThreadDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !isSubmitting) {
          onClose();
        }
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogTitle>Rename chat</DialogTitle>
        <DialogDescription>
          Update the chat title shown in your history.
        </DialogDescription>
        <Input
          value={value}
          onChange={(event) => onValueChange(event.target.value)}
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
          disabled={isSubmitting}
          maxLength={maxLength}
          autoFocus
        />
        {requestError ? (
          <p className="text-sm text-destructive">{requestError}</p>
        ) : validationError ? (
          <p className="text-sm text-destructive">{validationError}</p>
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
          <Button type="button" onClick={onSubmit} disabled={!canSubmit}>
            {isSubmitting ? "Saving..." : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
