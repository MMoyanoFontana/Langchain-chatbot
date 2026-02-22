"use client"

import { useCallback, useEffect, useRef } from "react"

type ShortcutOptions = {
  disabled?: boolean
  ignoreInput?: boolean
  preventDefault?: boolean
}

const MODIFIER_ALIASES = new Set([
  "ctrl",
  "control",
  "meta",
  "cmd",
  "command",
  "shift",
  "alt",
  "option",
])

const isPeriodKey = (event: KeyboardEvent) =>
  event.code === "Period" || event.key === "." || event.key === ">"

const isCommaKey = (event: KeyboardEvent) =>
  event.code === "Comma" || event.key === "," || event.key === "<"

const SPECIAL_KEYS: Record<string, (event: KeyboardEvent) => boolean> = {
  ".": isPeriodKey,
  period: isPeriodKey,
  dot: isPeriodKey,
  ",": isCommaKey,
  comma: isCommaKey,
}

function isEditableElement(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false
  }

  if (target.isContentEditable) {
    return true
  }

  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
}

function matchPrimaryKey(token: string, event: KeyboardEvent) {
  const normalizedKey = event.key.toLowerCase()
  const normalizedCode = event.code.toLowerCase()

  const specialMatcher = SPECIAL_KEYS[token]
  if (specialMatcher) {
    return specialMatcher(event)
  }

  return (
    normalizedKey === token ||
    normalizedCode === token ||
    normalizedCode === `key${token}` ||
    normalizedCode === `digit${token}`
  )
}

export function useKeyboardShortcut(
  keys: string[],
  callback: () => void,
  options: ShortcutOptions = {}
) {
  const { disabled = false, ignoreInput = true, preventDefault = true } = options
  const callbackRef = useRef(callback)

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (disabled) {
        return
      }

      if (ignoreInput && isEditableElement(event.target)) {
        return
      }

      const normalizedKeys = keys.map((key) => key.trim().toLowerCase())
      const hasCtrl = normalizedKeys.includes("ctrl") || normalizedKeys.includes("control")
      const hasMeta = normalizedKeys.includes("meta") || normalizedKeys.includes("cmd") || normalizedKeys.includes("command")
      const hasShift = normalizedKeys.includes("shift")
      const hasAlt = normalizedKeys.includes("alt") || normalizedKeys.includes("option")

      if (event.ctrlKey !== hasCtrl) return
      if (event.metaKey !== hasMeta) return
      if (event.shiftKey !== hasShift) return
      if (event.altKey !== hasAlt) return

      const nonModifierKeys = normalizedKeys.filter((key) => !MODIFIER_ALIASES.has(key))
      const keyMatched =
        nonModifierKeys.length === 0 ||
        nonModifierKeys.some((key) => matchPrimaryKey(key, event))

      if (!keyMatched) {
        return
      }

      if (preventDefault) {
        event.preventDefault()
      }

      callbackRef.current()
    },
    [disabled, ignoreInput, keys, preventDefault]
  )

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [handleKeyDown])
}
