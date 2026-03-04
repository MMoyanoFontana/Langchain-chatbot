"use client";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";
import type { AttachmentData } from "@/components/ai-elements/attachments";

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
    PromptInputBody,
    PromptInputButton,
    PromptInputFooter,
    PromptInputProvider,
    PromptInputSubmit,
    PromptInputTextarea,
    PromptInputTools,
    usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
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
}

interface ProviderModelGroup {
    providerLabel: string;
    providerSlug: string;
    models: ComposerModel[];
}

const getProviderGroupKey = (group: ProviderModelGroup): string =>
    `${group.providerSlug}:${group.providerLabel}`;

const PROVIDER_LOGO_MAP: Record<string, string> = {
    gemini: "google",
};

const FALLBACK_MODELS: ComposerModel[] = [
    {
        id: "gpt-4o",
        name: "GPT-4o",
        providerLabel: "OpenAI",
        providerSlug: "openai",
    },
];

const normalizeProviderSlug = (providerCode: string): string =>
    PROVIDER_LOGO_MAP[providerCode] ?? providerCode;

const mapCatalogModelToComposerModel = (
    catalogModel: CatalogModelResponse
): ComposerModel => {
    const providerSlug = normalizeProviderSlug(catalogModel.provider.code);

    return {
        id: catalogModel.model_id,
        name: catalogModel.display_name,
        providerLabel: catalogModel.provider.display_name,
        providerSlug,
    };
};

const SUBMITTING_TIMEOUT = 200;
const STREAMING_TIMEOUT = 2000;

interface AttachmentItemProps {
    attachment: Extract<AttachmentData, { type: "file" }>;
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
            <AttachmentRemove />
        </Attachment>
    );
});

AttachmentItem.displayName = "AttachmentItem";

interface ModelItemProps {
    m: ComposerModel;
    selectedModel: string;
    onSelect: (id: string) => void;
    className?: string;
}

const ModelItem = memo(({ m, selectedModel, onSelect, className }: ModelItemProps) => {
    const handleSelect = useCallback(() => onSelect(m.id), [onSelect, m.id]);
    return (
        <ModelSelectorItem
            className={className}
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
        <Attachments variant="inline">
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
    onSubmitMessage?: (message: PromptInputMessage) => void | Promise<void>;
}

const PromptComposer = ({ className, onSubmitMessage }: PromptComposerProps) => {
    const [availableModels, setAvailableModels] = useState<ComposerModel[]>(FALLBACK_MODELS);
    const [model, setModel] = useState<string>(FALLBACK_MODELS[0].id);
    const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
    const [collapsedProviders, setCollapsedProviders] = useState<Set<string>>(new Set());
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
                    setAvailableModels(FALLBACK_MODELS);
                    setModel(FALLBACK_MODELS[0].id);
                    return;
                }

                setAvailableModels(mappedModels);
                setModel((previous) =>
                    mappedModels.some((entry) => entry.id === previous)
                        ? previous
                        : mappedModels[0].id
                );
            } catch {
                if (!isCancelled) {
                    setAvailableModels(FALLBACK_MODELS);
                    setModel(FALLBACK_MODELS[0].id);
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

    const selectedModelData = availableModels.find((entry) => entry.id === model);

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
                    models: [entry],
                });
            }
        }

        return Array.from(grouped.values());
    }, [availableModels]);

    const handleModelSelect = useCallback((id: string) => {
        setModel(id);
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

        setStatus("submitted");
        setStatus("streaming");

        const submitPromise = onSubmitMessage?.(message);
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
    }, [onSubmitMessage]);

    return (
        <div className={className ?? "size-full"}>
            <PromptInputProvider>
                <PromptInput globalDrop multiple onSubmit={handleSubmit}>
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
                                        <PromptInputButton>
                                            {selectedModelData?.providerSlug && (
                                                <ModelSelectorLogo
                                                    provider={selectedModelData.providerSlug}
                                                />
                                            )}
                                            <ModelSelectorName>
                                                {selectedModelData?.name ?? "Select model"}
                                            </ModelSelectorName>
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
                                            const isCollapsed = collapsedProviders.has(groupKey);

                                            return (
                                                <ModelSelectorGroup
                                                    heading={
                                                        <button
                                                            aria-label={`Toggle ${group.providerLabel} models`}
                                                            className="inline-flex w-full cursor-pointer items-center gap-2 rounded-sm px-1 py-0.5 text-left text-sm font-semibold transition-colors hover:bg-muted"
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
                                                    }
                                                    key={groupKey}
                                                >
                                                    {group.models.map((entry) => (
                                                        <ModelItem
                                                            className={isCollapsed ? "hidden" : undefined}
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
                        <PromptInputSubmit status={status} />
                    </PromptInputFooter>
                </PromptInput>
            </PromptInputProvider>
        </div>
    );
};

export default PromptComposer;
