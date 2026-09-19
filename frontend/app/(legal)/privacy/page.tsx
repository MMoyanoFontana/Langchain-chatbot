import type { Metadata } from "next"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Privacy Policy — Langchain Chatbot",
  description:
    "How this personal portfolio demo collects, uses, and deletes your data.",
}

const LAST_UPDATED = "19 September 2026"
const CONTACT_EMAIL = "mmoyanofontana@gmail.com"

export default function PrivacyPage() {
  return (
    <article className="prose-legal">
      <h1 className="text-2xl font-semibold tracking-tight">Privacy Policy</h1>
      <p className="text-muted-foreground mt-1 text-sm">
        Last updated: {LAST_UPDATED}
      </p>

      <section className="border-muted-foreground/20 bg-muted/40 mt-6 rounded-lg border p-4 text-sm">
        <p className="font-medium">In short</p>
        <p className="text-muted-foreground mt-2">
          This is a personal portfolio demo, not a commercial product. Please do
          not upload confidential, sensitive, or regulated personal information.
          Everything you send is stored in a database and forwarded to the AI
          provider you select. You can delete your account and all associated
          data at any time from Settings.
        </p>
      </section>

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
          <strong>Account data</strong> — your username, and optionally a display
          name, email address, and avatar URL. If you sign in with Google,
          GitHub, or Microsoft, the Demo receives your basic profile and email
          address from that provider (scopes: <code>openid</code>,{" "}
          <code>email</code>, <code>profile</code> or equivalent) and stores a
          provider identifier to link the account.
        </li>
        <li>
          <strong>Credentials</strong> — if you register with a password, only a
          salted hash is stored. The plaintext password is never retained.
        </li>
        <li>
          <strong>Chat content</strong> — your messages, the model&rsquo;s
          replies, per-thread system prompts, thread titles, and rolling
          conversation summaries.
        </li>
        <li>
          <strong>Uploaded files</strong> — files you attach are parsed into text
          chunks, converted to vector embeddings, and stored in a Pinecone index
          under a namespace scoped to your user and thread. Filenames, media
          types, sizes, and checksums are stored in the database.
        </li>
        <li>
          <strong>Memory</strong> — facts the assistant extracts from your
          conversations in order to personalise later replies. These are visible
          and individually deletable under Settings &rarr; Memory.
        </li>
        <li>
          <strong>Provider API keys</strong> — if you supply your own API keys,
          they are encrypted at rest (Fernet symmetric encryption) before being
          written to the database, and are decrypted only to make requests on
          your behalf.
        </li>
        <li>
          <strong>Technical metadata</strong> — session records, token counts,
          latency measurements, and timestamps.
        </li>
      </ul>
      <p>
        There is no analytics, advertising, tracking pixel, or third-party
        behavioural profiling in the Demo.
      </p>

      <h2>3. Cookies</h2>
      <p>
        The Demo sets a strictly necessary, <code>HttpOnly</code> session cookie
        to keep you signed in, and a short-lived nonce cookie during OAuth sign-in
        to protect against cross-site request forgery. Both are essential to the
        service, so no consent banner is presented. No advertising or analytics
        cookies are set.
      </p>

      <h2>4. Why it is processed, and on what legal basis</h2>
      <ul>
        <li>
          <strong>To provide the service</strong> (authenticating you, storing
          your threads, generating replies) — performance of a contract, or your
          consent where no contract exists.
        </li>
        <li>
          <strong>To keep the service secure and available</strong> (session
          management, rate limiting, abuse prevention) — legitimate interests.
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
          <strong>Pinecone</strong> — vector storage for document embeddings.
        </li>
        <li>
          <strong>Vercel</strong> — frontend hosting.
        </li>
        <li>
          <strong>Render / Railway</strong> — backend and PostgreSQL hosting.
        </li>
        <li>
          <strong>LangSmith</strong> — optional request tracing, when enabled.
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
        a thread removes its messages and the associated vectors. Deleting your
        account removes your profile, all threads and messages, stored memories,
        encrypted API keys, and indexed documents. Because this is a demo, the
        entire dataset may be wiped at any time without notice.
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
          <strong>Access and portability</strong> — export any thread as Markdown
          or JSON from the thread menu.
        </li>
        <li>
          <strong>Erasure</strong> — delete individual messages, threads,
          documents, and memories, or delete your entire account under Settings
          &rarr; Profile.
        </li>
      </ul>
      <p>
        For anything not covered by those controls, email{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>. As a one-person
        project, responses are on a best-effort basis.
      </p>

      <h2>9. Security</h2>
      <p>
        Passwords are hashed, provider API keys are encrypted at rest, session
        cookies are <code>HttpOnly</code>, and access to threads and documents is
        scoped per user. That said, no system is perfectly secure, and this Demo
        has not undergone a third-party security audit. Do not store anything you
        could not afford to have exposed.
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
