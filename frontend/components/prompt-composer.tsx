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
import type { BackendProviderCode, SubmitMessagePayload } from "@/contexts/chat-composer-context";
import { CheckIcon, ChevronDown, ChevronRight } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useState } from "react";

interface CatalogProvider {
    id: number;
    code: string;
    display_name: string;
    is_active: boolean;
}

interface CatalogModelResponse {
    id: number;
    model_id: string;
    display_name: string;
    is_active: boolean;
    provider: CatalogProvider;
}

interface ComposerModel {
    id: string;
    name: string;
    providerLabel: string;
    providerSlug: string;
    providerCode: BackendProviderCode;
}

interface ProviderModelGroup {
    providerLabel: string;
    providerSlug: string;
    providerCode: BackendProviderCode;
    models: ComposerModel[];
}

const getProviderGroupKey = (group: ProviderModelGroup): string =>
    `${group.providerSlug}:${group.providerLabel}`;

const PROVIDER_LOGO_MAP: Record<string, string> = {
    gemini: "google",
};

const normalizeProviderSlug = (providerCode: string): string =>
    PROVIDER_LOGO_MAP[providerCode] ?? providerCode;

const normalizeProviderCode = (providerCode: string): BackendProviderCode => {
    if (
        providerCode === "openai" ||
        providerCode === "anthropic" ||
        providerCode === "gemini" ||
        providerCode === "groq" ||
        providerCode === "other"
    ) {
        return providerCode;
    }
    return "other";
};

const mapCatalogModelToComposerModel = (
    catalogModel: CatalogModelResponse
): ComposerModel => {
    const providerSlug = normalizeProviderSlug(catalogModel.provider.code);

    return {
        id: catalogModel.model_id,
        name: catalogModel.display_name,
        providerLabel: catalogModel.provider.display_name,
        providerSlug,
        providerCode: normalizeProviderCode(catalogModel.provider.code),
    };
};

const SUBMITTING_TIMEOUT = 200;
const STREAMING_TIMEOUT = 2000;
const SELECTED_MODEL_STORAGE_KEY = "langchain-chatbot.selected-model.v1";

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
    onSubmitMessage?: (payload: SubmitMessagePayload) => void | Promise<void>;
}

interface ProviderApiKeyResponse {
    provider?: {
        code?: string;
    };
    is_active?: boolean;
}

const normalizeModelId = (value: string | null | undefined): string | null => {
    const normalized = value?.trim();
    return normalized || null;
};

const PromptComposer = ({
    className,
    preferredModelId,
    onSubmitMessage,
}: PromptComposerProps) => {
    const normalizedPreferredModelId = normalizeModelId(preferredModelId);
    const [availableModels, setAvailableModels] = useState<ComposerModel[]>([]);
    const [model, setModel] = useState<string | null>(
        () =>
            normalizedPreferredModelId ??
            readStoredSelectedModel() ??
            null
    );
    const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
    const [collapsedProviders, setCollapsedProviders] = useState<Set<string>>(new Set());
    const [connectedProviderCodes, setConnectedProviderCodes] = useState<Set<BackendProviderCode>>(new Set());
    const [modelsLoading, setModelsLoading] = useState(true);
    const [status, setStatus] = useState<
        "submitted" | "streaming" | "ready" | "error"
    >("ready");

    useEffect(() => {
        let isCancelled = false;

        const loadModels = async () => {
            try {
                const response = await fetch("/api/models", { cache: "no-store" });
                if (!response.ok) {
                    throw new Error(`Models request failed with status ${response.status}`);
                }

                const payload = (await response.json()) as CatalogModelResponse[];
                const mappedModels = payload.map(mapCatalogModelToComposerModel);

                if (isCancelled) {
                    return;
                }

                if (mappedModels.length === 0) {
                    setAvailableModels([]);
                    setModel(null);
                    return;
                }

                setAvailableModels(mappedModels);
                setModel((previous) =>
                    mappedModels.some((entry) => entry.id === previous)
                        ? previous
                        : null
                );
            } catch {
                if (!isCancelled) {
                    setAvailableModels([]);
                    setModel(null);
                }
            } finally {
                if (!isCancelled) {
                    setModelsLoading(false);
                }
            }
        };

        void loadModels();

        return () => {
            isCancelled = true;
        };
    }, []);

    useEffect(() => {
        if (!normalizedPreferredModelId) {
            return;
        }

        setModel((previous) =>
            previous === normalizedPreferredModelId ? previous : normalizedPreferredModelId
        );
    }, [normalizedPreferredModelId]);

    useEffect(() => {
        let isCancelled = false;

        const loadProviderKeys = async () => {
            try {
                const response = await fetch("/api/user/provider-keys", { cache: "no-store" });
                if (!response.ok) {
                    throw new Error(`Provider keys request failed with status ${response.status}`);
                }

                const payload = (await response.json()) as ProviderApiKeyResponse[];
                if (isCancelled) {
                    return;
                }

                const connectedCodes = new Set<BackendProviderCode>();
                for (const entry of payload) {
                    if (entry.is_active === false) {
                        continue;
                    }
                    const providerCode = normalizeProviderCode(
                        entry.provider?.code?.toLowerCase() ?? ""
                    );
                    connectedCodes.add(providerCode);
                }

                setConnectedProviderCodes(connectedCodes);
            } catch {
                if (!isCancelled) {
                    setConnectedProviderCodes(new Set());
                }
            }
        };

        void loadProviderKeys();

        return () => {
            isCancelled = true;
        };
    }, []);

    useEffect(() => {
        if (modelsLoading) {
            return;
        }

        const connectedModels = availableModels.filter((entry) =>
            connectedProviderCodes.has(entry.providerCode)
        );

        setModel((previous) => {
            if (previous && connectedModels.some((entry) => entry.id === previous)) {
                return previous;
            }

            if (
                normalizedPreferredModelId &&
                connectedModels.some((entry) => entry.id === normalizedPreferredModelId)
            ) {
                return normalizedPreferredModelId;
            }

            if (connectedModels.length === 0) {
                return null;
            }

            return connectedModels[0].id;
        });
    }, [availableModels, connectedProviderCodes, modelsLoading, normalizedPreferredModelId]);

    useEffect(() => {
        if (model) {
            persistSelectedModel(model);
            return;
        }

        clearStoredSelectedModel();
    }, [model]);

    const selectedModelData =
        model ? availableModels.find((entry) => entry.id === model) ?? null : null;
    const canSubmit = Boolean(
        selectedModelData &&
        connectedProviderCodes.has(selectedModelData.providerCode)
    );

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
        setModel(id);
        persistSelectedModel(id);
        setModelSelectorOpen(false);
    }, []);

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
        if (!canSubmit || !selectedModelData) {
            return;
        }

        setStatus("submitted");
        setStatus("streaming");

        const submitPromise = onSubmitMessage?.({
            message,
            modelId: selectedModelData.id,
            providerCode: selectedModelData.providerCode,
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
    }, [canSubmit, onSubmitMessage, selectedModelData]);

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
                                                    {group.models.map((entry) => (
                                                        <ModelItem
                                                            className={isCollapsed ? "hidden" : undefined}
                                                            disabled={!providerConnected}
                                                            key={entry.id}
                                                            m={entry}
                                                            onSelect={handleModelSelect}
                                                            selectedModel={model}
                                                        />
                                                    ))}
                                                </ModelSelectorGroup>
                                            );
                                        })}
                                    </ModelSelectorList>
                                </ModelSelectorContent>
                            </ModelSelector>
                        </PromptInputTools>
                        <PromptInputSubmit
                            disabled={modelsLoading || !canSubmit}
                            status={status}
                        />
                    </PromptInputFooter>
                </PromptInput>
        </div>
    );
};

export default PromptComposer;
