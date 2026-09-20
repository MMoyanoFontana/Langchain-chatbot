import type { Metadata } from "next"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Privacy Policy | Langchain Chatbot",
  description:
    "How this personal portfolio demo collects, uses, and deletes your data.",
}

const LAST_UPDATED = "20 September 2026"
const CONTACT_EMAIL = "mmoyanofontana@gmail.com"

export default function PrivacyPage() {
  return (
    <article className="prose-legal">
      <h1 className="text-2xl font-semibold tracking-tight">Privacy Policy</h1>
      <p className="text-muted-foreground mt-1 text-sm">
        Last updated: {LAST_UPDATED}
      </p>

      <h2>1. Who is responsible for your data</h2>
      <p>
        This project (&ldquo;the Demo&rdquo;) is operated by an individual
        developer as a non-commercial portfolio project. For the purposes of the
        EU/UK General Data Protection Regulation (GDPR), that individual is the
        data controller and can be reached at{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
      </p>

      <h2>2. What data is collected</h2>
      <p>The Demo stores only what it needs in order to work:</p>
      <ul>
        <li>
          <strong>Account data</strong>: your username, and optionally a display
          name, email address, and avatar picture. If you sign in with Google,
          GitHub, or Microsoft, the Demo receives your basic profile and email
          address from that provider, and stores an identifier so the account can
          be linked on your next sign-in.
        </li>
        <li>
          <strong>Credentials</strong>: if you register with a password, it is
          stored only in an encrypted form. Your actual password is never
          retained and cannot be recovered.
        </li>
        <li>
          <strong>Chat content</strong>: your messages, the assistant&rsquo;s
          replies, any custom instructions you set for a conversation,
          conversation titles, and summaries of your conversations.
        </li>
        <li>
          <strong>Uploaded files</strong>: files you attach are processed so the
          assistant can search them, and stored so they remain available to the
          conversation you attached them to. Basic details about each file, such
          as its name, type, and size, are stored alongside it.
        </li>
        <li>
          <strong>Memory</strong>: facts the assistant saves from your
          conversations in order to personalise later replies. These are visible
          and individually deletable under Settings &rarr; Memory.
        </li>
        <li>
          <strong>Provider API keys</strong>: if you supply your own API keys,
          they are encrypted before being stored, and are used only to make
          requests to that provider on your behalf.
        </li>
        <li>
          <strong>Technical metadata</strong>: sign-in session records, basic
          usage and performance measurements, and timestamps.
        </li>
      </ul>
      <p>
        There is no analytics, advertising, tracking pixel, or third-party
        behavioural profiling in the Demo.
      </p>

      <h2>3. Cookies</h2>
      <p>
        The Demo sets a strictly necessary cookie to keep you signed in, and a
        short-lived cookie during sign-in with Google, GitHub, or Microsoft to
        protect that process against abuse. Both are essential to the service, so
        no consent banner is presented. No advertising or analytics cookies are
        set.
      </p>

      <h2>4. Why it is processed, and on what legal basis</h2>
      <ul>
        <li>
          <strong>To provide the service</strong> (signing you in, storing your
          conversations, generating replies): performance of a contract, or your
          consent where no contract exists.
        </li>
        <li>
          <strong>To keep the service secure and available</strong> (managing
          sign-in sessions, preventing abuse): legitimate interests.
        </li>
      </ul>

      <h2>5. Who your data is shared with</h2>
      <p>
        Your prompts, attached document excerpts, and conversation context are
        transmitted to the AI provider you choose for that message. Depending on
        your selection that may be OpenAI, Anthropic, Google, Groq, or a local
        Ollama instance. Their own privacy terms govern what they do with that
        data, and this policy cannot extend to them.
      </p>
      <p>Other infrastructure providers used by the Demo:</p>
      <ul>
        <li>
          <strong>Pinecone</strong>: storage for uploaded document content.
        </li>
        <li>
          <strong>Vercel</strong>: application hosting.
        </li>
        <li>
          <strong>Render / Railway</strong>: application and database hosting.
        </li>
        <li>
          <strong>LangSmith</strong>: optional diagnostics, when enabled.
        </li>
      </ul>
      <p>
        Your data is never sold, rented, or shared for advertising. It may be
        disclosed if required by law.
      </p>

      <h2>6. International transfers</h2>
      <p>
        The providers above operate infrastructure in various countries,
        including the United States. Using the Demo involves transferring your
        data outside the European Economic Area. Given the non-commercial nature
        of this project, no separate transfer mechanism has been negotiated with
        them beyond the standard terms each provider offers. Take this into
        account before entering personal data.
      </p>

      <h2>7. Retention</h2>
      <p>
        Data is kept until you delete it or until the Demo is shut down. Deleting
        a conversation removes its messages and anything stored from the files you
        attached to it. Deleting your account removes your profile, all
        conversations and messages, saved memories, stored API keys, and uploaded
        documents. Because this is a demo, all stored data may be removed at any
        time without notice.
      </p>

      <h2>8. Your rights</h2>
      <p>
        Under GDPR and comparable laws you have the right to access, correct,
        export, and erase your data, to restrict or object to processing, and to
        lodge a complaint with your local supervisory authority. The Demo
        implements the key ones directly in the product:
      </p>
      <ul>
        <li>
          <strong>Access and portability</strong>: you can export any
          conversation from its menu, in a choice of common file formats.
        </li>
        <li>
          <strong>Erasure</strong>: you can delete individual messages,
          conversations, documents, and memories, or delete your entire account
          under Settings &rarr; Profile.
        </li>
      </ul>
      <p>
        For anything not covered by those controls, email{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>. As a one-person
        project, responses are on a best-effort basis.
      </p>

      <h2>9. Security</h2>
      <p>
        Passwords and provider API keys are stored in an encrypted form, sign-in
        sessions are protected, and your conversations and documents are
        accessible only to your own account. That said, no system is perfectly
        secure, and this Demo has not undergone a third-party security audit. Do
        not store anything you could not afford to have exposed.
      </p>

      <h2>10. Children</h2>
      <p>
        The Demo is not intended for anyone under 16, and data from children is
        not knowingly collected. If you believe a child has provided data, contact
        the address above and it will be deleted.
      </p>

      <h2>11. Changes</h2>
      <p>
        This policy may be updated; the date at the top reflects the latest
        version. Material changes will be surfaced in the app where practical.
      </p>

      <h2>12. Contact</h2>
      <p>
        Questions or data requests:{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>. See also the{" "}
        <Link href="/terms">Terms of Use</Link>.
      </p>
    </article>
  )
}
