"use client";

import {
    Attachment,
    AttachmentPreview,
    AttachmentRemove,
    Attachments,
} from "@/components/ai-elements/attachments";
import {
    ModelSelector,
    ModelSelectorContent,
    ModelSelectorEmpty,
    ModelSelectorGroup,
    ModelSelectorInput,
    ModelSelectorItem,
    ModelSelectorList,
    ModelSelectorLogo,
    ModelSelectorName,
    ModelSelectorTrigger,
} from "@/components/ai-elements/model-selector";
import {
    PromptInput,
    PromptInputActionAddAttachments,
    PromptInputActionMenu,
    PromptInputActionMenuContent,
    PromptInputActionMenuTrigger,
    type PromptInputAttachment,
    PromptInputBody,
    PromptInputButton,
    PromptInputFooter,
    type PromptInputMessage,
    PromptInputSubmit,
    PromptInputTextarea,
    PromptInputTools,
    usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type {
  BackendProviderCode,
  CompareModelSelection,
  SubmitMessagePayload,
} from "@/contexts/chat-composer-context";
import { useChatModels, type ChatModel } from "@/lib/use-chat-models";
import { CheckIcon, ChevronDown, ChevronRight, GitCompareIcon } from "lucide-react";
import { useCurrentUser } from "@/contexts/user-context";
import { memo, useCallback, useEffect, useMemo, useState } from "react";

type ComposerModel = ChatModel;

interface ProviderModelGroup {
    providerLabel: string;
    providerSlug: string;
    providerCode: BackendProviderCode;
    models: ComposerModel[];
}

const getProviderGroupKey = (group: ProviderModelGroup): string =>
    `${group.providerSlug}:${group.providerLabel}`;

const SUBMITTING_TIMEOUT = 200;
const STREAMING_TIMEOUT = 2000;
const SELECTED_MODEL_STORAGE_KEY = "langchain-chatbot.selected-model.v1";
const COMPARE_MAX_MODELS = 3;

const readStoredSelectedModel = (): string | null => {
    if (typeof window === "undefined") {
        return null;
    }
    try {
        const storedValue = window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY);
        return storedValue?.trim() || null;
    } catch {
        return null;
    }
};

const persistSelectedModel = (modelId: string) => {
    if (typeof window === "undefined") {
        return;
    }
    try {
        window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, modelId);
    } catch {
        // Ignore storage errors.
    }
};

const clearStoredSelectedModel = () => {
    if (typeof window === "undefined") {
        return;
    }
    try {
        window.localStorage.removeItem(SELECTED_MODEL_STORAGE_KEY);
    } catch {
        // Ignore storage errors.
    }
};

interface AttachmentItemProps {
    attachment: PromptInputAttachment;
    onRemove: (id: string) => void;
}

const AttachmentItem = memo(({ attachment, onRemove }: AttachmentItemProps) => {
    const handleRemove = useCallback(
        () => onRemove(attachment.id),
        [onRemove, attachment.id]
    );
    return (
        <Attachment data={attachment} key={attachment.id} onRemove={handleRemove}>
            <AttachmentPreview />
            <div className="min-w-0 flex-1">
                <div className="truncate">{attachment.filename ?? "Attachment"}</div>
                <span className="block truncate text-muted-foreground text-xs">
                    {attachment.mediaType}
                </span>
            </div>
            <AttachmentRemove />
        </Attachment>
    );
});

AttachmentItem.displayName = "AttachmentItem";

interface ModelItemProps {
    m: ComposerModel;
    selectedModel: string | null;
    onSelect: (id: string) => void;
    disabled?: boolean;
    className?: string;
}

const ModelItem = memo(({ m, selectedModel, onSelect, disabled, className }: ModelItemProps) => {
    const handleSelect = useCallback(() => {
        if (disabled) {
            return;
        }
        onSelect(m.id);
    }, [disabled, onSelect, m.id]);
    return (
        <ModelSelectorItem
            className={className}
            disabled={disabled}
            key={m.id}
            onSelect={handleSelect}
            value={m.id}
        >
            <ModelSelectorName>{m.name}</ModelSelectorName>
            {selectedModel === m.id ? (
                <CheckIcon className="ml-auto size-4" />
            ) : (
                <div className="ml-auto size-4" />
            )}
        </ModelSelectorItem>
    );
});

ModelItem.displayName = "ModelItem";

const PromptInputAttachmentsDisplay = () => {
    const attachments = usePromptInputAttachments();

    const handleRemove = useCallback(
        (id: string) => attachments.remove(id),
        [attachments]
    );

    if (attachments.files.length === 0) {
        return null;
    }

    return (
        <Attachments
            className="w-full self-stretch justify-start px-2.5 pt-2.5"
            variant="inline"
        >
            {attachments.files.map((attachment) => (
                <AttachmentItem
                    attachment={attachment}
                    key={attachment.id}
                    onRemove={handleRemove}
                />
            ))}
        </Attachments>
    );
};

interface PromptComposerProps {
    className?: string;
    preferredModelId?: string | null;
    forceModelId?: string | null;
    onSubmitMessage?: (payload: SubmitMessagePayload) => void | Promise<void>;
}

const normalizeModelId = (value: string | null | undefined): string | null => {
    const normalized = value?.trim();
    return normalized || null;
};

const PromptComposer = ({
    className,
    preferredModelId,
    forceModelId,
    onSubmitMessage,
}: PromptComposerProps) => {
    const { user } = useCurrentUser();
    const normalizedPreferredModelId = normalizeModelId(preferredModelId);
    const {
        models: availableModels,
        connectedProviderCodes,
        loading: modelsLoading,
    } = useChatModels();
    // The user's explicit pick. `null` means "fall back to defaults".
    // `effectiveModel` below derives the actual id used by the UI.
    const [userSelectedModel, setUserSelectedModel] = useState<string | null>(
        () => readStoredSelectedModel()
    );
    const [compareEnabled, setCompareEnabled] = useState(false);
    const [compareModels, setCompareModels] = useState<string[]>([]);
    const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
    const [collapsedProviders, setCollapsedProviders] = useState<Set<string>>(new Set());
    const [status, setStatus] = useState<
        "submitted" | "streaming" | "ready" | "error"
    >("ready");

    // Derived effective model id: prefers the user's pick when valid, else
    // the parent-provided preferred model, else the first connected model.
    // Computed (not state-synced) so React stays in sync without effects.
    const effectiveModel = useMemo<string | null>(() => {
        if (modelsLoading) return userSelectedModel;
        const isValid = (id: string | null): boolean => {
            if (!id) return false;
            const entry = availableModels.find((m) => m.id === id);
            return Boolean(entry && connectedProviderCodes.has(entry.providerCode));
        };
        if (isValid(userSelectedModel)) return userSelectedModel;
        if (isValid(normalizedPreferredModelId)) return normalizedPreferredModelId;
        const firstConnected = availableModels.find((entry) =>
            connectedProviderCodes.has(entry.providerCode)
        );
        return firstConnected?.id ?? null;
    }, [
        availableModels,
        connectedProviderCodes,
        modelsLoading,
        normalizedPreferredModelId,
        userSelectedModel,
    ]);

    // When a winner is selected from comparison mode, override the user's model choice.
    const normalizedForceModelId = normalizeModelId(forceModelId);
    useEffect(() => {
        if (!normalizedForceModelId) return;
        setUserSelectedModel(normalizedForceModelId);
    }, [normalizedForceModelId]);

    // Persist whichever model is currently effective so it survives reloads.
    // Only writes to storage — no setState — so this is a safe effect.
    useEffect(() => {
        if (effectiveModel) {
            persistSelectedModel(effectiveModel);
        } else {
            clearStoredSelectedModel();
        }
    }, [effectiveModel]);

    const selectedModelData =
        effectiveModel
            ? availableModels.find((entry) => entry.id === effectiveModel) ?? null
            : null;
    const compareModelsValid = compareModels
        .map((id) => availableModels.find((entry) => entry.id === id))
        .filter((entry): entry is ComposerModel =>
            Boolean(entry) && connectedProviderCodes.has(entry!.providerCode)
        );
    const canSubmit = compareEnabled
        ? compareModelsValid.length >= 1
        : Boolean(
              selectedModelData &&
              connectedProviderCodes.has(selectedModelData.providerCode)
          );
    const compareSelectionLabel =
        compareEnabled && compareModelsValid.length > 0
            ? compareModelsValid.length === 1
                ? compareModelsValid[0].name
                : `${compareModelsValid.length} models`
            : null;

    const modelsByProvider = useMemo(() => {
        const grouped = new Map<string, ProviderModelGroup>();

        for (const entry of availableModels) {
            const existingGroup = grouped.get(entry.providerLabel);
            if (existingGroup) {
                existingGroup.models.push(entry);
            } else {
                grouped.set(entry.providerLabel, {
                    providerLabel: entry.providerLabel,
                    providerSlug: entry.providerSlug,
                    providerCode: entry.providerCode,
                    models: [entry],
                });
            }
        }

        return Array.from(grouped.values()).sort((a, b) => {
            const aConnected = connectedProviderCodes.has(a.providerCode);
            const bConnected = connectedProviderCodes.has(b.providerCode);
            if (aConnected === bConnected) {
                return a.providerLabel.localeCompare(b.providerLabel);
            }
            return aConnected ? -1 : 1;
        });
    }, [availableModels, connectedProviderCodes]);

    const handleModelSelect = useCallback((id: string) => {
        if (compareEnabled) {
            setCompareModels((prev) => {
                if (prev.includes(id)) {
                    return prev.filter((entry) => entry !== id);
                }
                if (prev.length >= COMPARE_MAX_MODELS) {
                    return prev;
                }
                return [...prev, id];
            });
            return;
        }
        setUserSelectedModel(id);
        setModelSelectorOpen(false);
    }, [compareEnabled]);

    const toggleCompareMode = useCallback(() => {
        setCompareEnabled((prev) => {
            const next = !prev;
            if (next) {
                // seed with the currently selected model
                setCompareModels(effectiveModel ? [effectiveModel] : []);
            } else {
                setCompareModels([]);
            }
            return next;
        });
    }, [effectiveModel]);

    const toggleProviderGroup = useCallback((groupKey: string) => {
        setCollapsedProviders((previous) => {
            const next = new Set(previous);
            if (next.has(groupKey)) {
                next.delete(groupKey);
            } else {
                next.add(groupKey);
            }
            return next;
        });
    }, []);

    const handleSubmit = useCallback((message: PromptInputMessage) => {
        const hasText = Boolean(message.text);
        const hasAttachments = Boolean(message.files?.length);

        if (!(hasText || hasAttachments)) {
            return;
        }

        let leadModel: ComposerModel | null = selectedModelData;
        let compareWith: CompareModelSelection[] | undefined;

        if (compareEnabled) {
            const compareModelData = compareModels
                .map((id) => availableModels.find((entry) => entry.id === id))
                .filter((entry): entry is ComposerModel =>
                    Boolean(entry) &&
                    connectedProviderCodes.has(entry!.providerCode)
                );
            if (compareModelData.length === 0) {
                return;
            }
            leadModel = compareModelData[0];
            compareWith = compareModelData.slice(1).map((entry) => ({
                modelId: entry.id,
                providerCode: entry.providerCode,
            }));
        }

        if (!leadModel || !connectedProviderCodes.has(leadModel.providerCode)) {
            return;
        }

        setStatus("submitted");
        setStatus("streaming");

        const submitPromise = onSubmitMessage?.({
            message,
            modelId: leadModel.id,
            providerCode: leadModel.providerCode,
            compareWith,
        });
        if (submitPromise instanceof Promise) {
            submitPromise
                .then(() => {
                    setStatus("ready");
                })
                .catch(() => {
                    setStatus("error");
                });
            return;
        }

        if (onSubmitMessage) {
            setStatus("ready");
            return;
        }

        setTimeout(() => {
            setStatus("ready");
        }, SUBMITTING_TIMEOUT + STREAMING_TIMEOUT);
    }, [
        availableModels,
        compareEnabled,
        compareModels,
        connectedProviderCodes,
        onSubmitMessage,
        selectedModelData,
    ]);

    if (!user) {
        return (
            <div className={`${className ?? "size-full"} flex items-center justify-center p-4`}>
                <a
                    href="/login"
                    className="inline-flex h-9 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
                >
                    Sign in to start chatting
                </a>
            </div>
        );
    }

    return (
        <div className={className ?? "size-full"}>
                <PromptInput
                    globalDrop
                    multiple
                    onSubmit={handleSubmit}
                >
                    <PromptInputAttachmentsDisplay />
                    <PromptInputBody>
                        <PromptInputTextarea
                            placeholder="Ask me anything..."
                            className="pt-3 placeholder:font-medium placeholder:tracking-[0.01em] focus:placeholder:text-muted-foreground/35 transition-colors"
                        />
                    </PromptInputBody>
                    <PromptInputFooter>
                        <PromptInputTools>
                            <PromptInputActionMenu>
                                <PromptInputActionMenuTrigger />
                                <PromptInputActionMenuContent>
                                    <PromptInputActionAddAttachments />
                                </PromptInputActionMenuContent>
                            </PromptInputActionMenu>
                            <PromptInputButton
                                disabled={modelsLoading}
                                onClick={toggleCompareMode}
                                size="sm"
                                variant={compareEnabled ? "default" : "ghost"}
                                title={
                                    compareEnabled
                                        ? "Compare mode: pick up to 3 models"
                                        : "Enable compare mode"
                                }
                            >
                                <GitCompareIcon className="size-4" />
                                <span>{compareEnabled ? "Compare" : "Compare"}</span>
                            </PromptInputButton>
                            <ModelSelector
                                onOpenChange={setModelSelectorOpen}
                                open={modelSelectorOpen}
                            >
                                <ModelSelectorTrigger
                                    render={
                                        <PromptInputButton disabled={modelsLoading} size="sm">
                                            {modelsLoading ? (
                                                <>
                                                    <Skeleton className="size-4 rounded-full" />
                                                    <Skeleton className="h-4 w-24" />
                                                </>
                                            ) : compareEnabled ? (
                                                <ModelSelectorName>
                                                    {compareSelectionLabel ?? "Pick models"}
                                                </ModelSelectorName>
                                            ) : (
                                                <>
                                                    {selectedModelData?.providerSlug && (
                                                        <ModelSelectorLogo
                                                            provider={selectedModelData.providerSlug}
                                                        />
                                                    )}
                                                    <ModelSelectorName>
                                                        {selectedModelData?.name ?? "No model selected"}
                                                    </ModelSelectorName>
                                                </>
                                            )}
                                        </PromptInputButton>
                                    }
                                />
                                <ModelSelectorContent>
                                    <ModelSelectorInput placeholder="Search models..." />
                                    <ModelSelectorList className="[scrollbar-width:thin] [scrollbar-color:var(--color-border)_transparent] [&::-webkit-scrollbar]:block [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border/70 hover:[&::-webkit-scrollbar-thumb]:bg-border">
                                        <ModelSelectorEmpty>
                                            {modelsLoading ? "Loading models..." : "No models found."}
                                        </ModelSelectorEmpty>
                                        {modelsByProvider.map((group) => {
                                            const groupKey = getProviderGroupKey(group);
                                            const providerConnected = connectedProviderCodes.has(group.providerCode);
                                            const isCollapsed = providerConnected
                                                ? collapsedProviders.has(groupKey)
                                                : true;

                                            return (
                                                <ModelSelectorGroup
                                                    heading={
                                                        providerConnected ? (
                                                            <button
                                                                aria-label={`Toggle ${group.providerLabel} models`}
                                                                className="inline-flex w-full cursor-pointer items-center gap-2 rounded-sm px-1 py-0.5 text-left text-sm font-semibold text-foreground transition-colors hover:bg-muted"
                                                                onClick={() => toggleProviderGroup(groupKey)}
                                                                type="button"
                                                            >
                                                                {isCollapsed ? (
                                                                    <ChevronRight className="size-4" />
                                                                ) : (
                                                                    <ChevronDown className="size-4" />
                                                                )}
                                                                <ModelSelectorLogo
                                                                    className="size-4"
                                                                    provider={group.providerSlug}
                                                                />
                                                                <span>{group.providerLabel}</span>
                                                            </button>
                                                        ) : (
                                                            <Tooltip>
                                                                <TooltipTrigger>
                                                                    <div
                                                                        aria-label={`${group.providerLabel} not connected`}
                                                                        className="inline-flex w-full cursor-default items-center gap-2 rounded-sm px-1 py-0.5 text-left text-sm font-semibold text-muted-foreground/70"
                                                                        role="note"
                                                                    >
                                                                        <ChevronRight className="size-4" />
                                                                        <ModelSelectorLogo
                                                                            className="size-4 opacity-70"
                                                                            provider={group.providerSlug}
                                                                        />
                                                                        <span>{group.providerLabel}</span>
                                                                    </div>
                                                                </TooltipTrigger>
                                                                <TooltipContent side="right">
                                                                    Not connected. Add API key in Settings.
                                                                </TooltipContent>
                                                            </Tooltip>
                                                        )
                                                    }
                                                    key={groupKey}
                                                >
                                                    {group.models.map((entry) => {
                                                        const isCompareSelected =
                                                            compareEnabled &&
                                                            compareModels.includes(entry.id);
                                                        const compareLimitReached =
                                                            compareEnabled &&
                                                            !isCompareSelected &&
                                                            compareModels.length >= COMPARE_MAX_MODELS;
                                                        const effectiveSelected = compareEnabled
                                                            ? isCompareSelected
                                                                ? entry.id
                                                                : null
                                                            : effectiveModel;
                                                        return (
                                                            <ModelItem
                                                                className={isCollapsed ? "hidden" : undefined}
                                                                disabled={
                                                                    !providerConnected || compareLimitReached
                                                                }
                                                                key={entry.id}
                                                                m={entry}
                                                                onSelect={handleModelSelect}
                                                                selectedModel={effectiveSelected}
                                                            />
                                                        );
                                                    })}
                                                </ModelSelectorGroup>
                                            );
                                        })}
                                    </ModelSelectorList>
                                </ModelSelectorContent>
                            </ModelSelector>
                        </PromptInputTools>
                        <Tooltip>
                            <TooltipTrigger render={<span />}>
                                <PromptInputSubmit
                                    disabled={modelsLoading || !canSubmit}
                                    status={status}
                                />
                            </TooltipTrigger>
                            {!modelsLoading && !canSubmit && (
                                <TooltipContent>
                                    Select a model to send a message
                                </TooltipContent>
                            )}
                        </Tooltip>
                    </PromptInputFooter>
                </PromptInput>
        </div>
    );
};

export default PromptComposer;
