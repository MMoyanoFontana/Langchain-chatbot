import type { Metadata } from "next"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Terms of Use | Langchain Chatbot",
  description:
    "The terms that apply to this personal portfolio demo, including disclaimers and limits of liability.",
}

const LAST_UPDATED = "20 September 2026"
const CONTACT_EMAIL = "mmoyanofontana@gmail.com"

export default function TermsPage() {
  return (
    <article className="prose-legal">
      <h1 className="text-2xl font-semibold tracking-tight">Terms of Use</h1>
      <p className="text-muted-foreground mt-1 text-sm">
        Last updated: {LAST_UPDATED}
      </p>

      <section className="border-muted-foreground/20 bg-muted/40 mt-6 rounded-lg border p-4 text-sm">
        <p className="font-medium">In short</p>
        <p className="text-muted-foreground mt-2">
          This is a free personal portfolio demo provided as-is, with no
          warranty, no uptime guarantee, and no support. AI output can be wrong,
          so do not rely on it. Your data may be deleted at any time. Use it at
          your own risk.
        </p>
      </section>

      <h2>1. What this is</h2>
      <p>
        Langchain Chatbot (&ldquo;the Demo&rdquo;) is a non-commercial project
        built by an individual developer to demonstrate technical work. It is not
        a product, not a business, and is offered free of charge. Nothing here
        creates a commercial relationship, a service-level commitment, or a
        professional engagement of any kind.
      </p>

      <h2>2. Eligibility</h2>
      <p>
        You must be at least 16 years old and legally able to agree to these
        terms. By using the Demo you confirm that you are.
      </p>

      <h2>3. Your account</h2>
      <p>
        You are responsible for keeping your credentials secure and for all
        activity that happens under your account. Do not impersonate anyone or
        register on someone else&rsquo;s behalf without their permission.
      </p>

      <h2>4. Acceptable use</h2>
      <p>You agree not to use the Demo to:</p>
      <ul>
        <li>break any applicable law or regulation;</li>
        <li>
          upload confidential information, trade secrets, health or financial
          records, government identifiers, or other sensitive or regulated
          personal data;
        </li>
        <li>
          upload content you do not have the rights to, or that infringes
          someone else&rsquo;s intellectual property or privacy;
        </li>
        <li>
          generate material that is unlawful, harassing, defamatory, or designed
          to harm others;
        </li>
        <li>
          attack, overload, probe, or reverse-engineer the service or its
          infrastructure, or circumvent usage limits and access controls;
        </li>
        <li>
          use the Demo for automated bulk processing or resell access to it.
        </li>
      </ul>
      <p>
        Accounts and content that breach these rules may be removed without
        notice.
      </p>

      <h2>5. Your content</h2>
      <p>
        You keep ownership of everything you submit. You grant only the narrow
        permission needed to run the service: to store your content, transmit it
        to the AI provider you select, index it for retrieval, and display it
        back to you. Your content is not used to train any models by the
        operator of this Demo, and it is not published or shared with other
        users.
      </p>
      <p>
        You are responsible for having the right to submit whatever you upload.
      </p>

      <h2>6. Third-party AI providers and API keys</h2>
      <p>
        The Demo routes your prompts to whichever provider you choose (OpenAI,
        Anthropic, Google, Groq, or a local model provider you configure). Your
        use of those models is also governed by that provider&rsquo;s own terms,
        and their behaviour, availability, and pricing are outside this
        project&rsquo;s control.
      </p>
      <p>
        If you supply your own API keys, you remain responsible for all charges
        your provider bills to them. Keys are stored in an encrypted form, but
        you supply them at your own risk, and you should use restricted keys with
        spending limits. No reimbursement is offered for any usage, overage, or
        compromise.
      </p>

      <h2>7. AI output and no reliance</h2>
      <p>
        Language models produce output that can be inaccurate, biased,
        incomplete, outdated, or entirely fabricated, including fabricated
        citations to your own uploaded documents. Output is generated
        automatically and is not reviewed by anyone.
      </p>
      <p>
        <strong>
          Nothing produced by the Demo is legal, medical, financial, or other
          professional advice.
        </strong>{" "}
        Do not rely on it for any decision that matters. Verify anything
        important against an authoritative source.
      </p>

      <h2>8. Availability</h2>
      <p>
        The Demo may be slow, broken, interrupted, changed, or discontinued at
        any time without notice. Stored data may be wiped during maintenance,
        migration, or shutdown. Keep your own copy of anything you want to
        keep. Conversations can be exported from the app.
      </p>

      <h2>9. Disclaimer of warranties</h2>
      <p className="uppercase">
        The Demo is provided &ldquo;as is&rdquo; and &ldquo;as available&rdquo;,
        without warranty of any kind, express or implied, including but not
        limited to the implied warranties of merchantability, fitness for a
        particular purpose, accuracy, and non-infringement. No warranty is given
        that the Demo will be uninterrupted, secure, or error-free.
      </p>

      <h2>10. Limitation of liability</h2>
      <p className="uppercase">
        To the fullest extent permitted by law, the operator of the Demo shall
        not be liable for any indirect, incidental, special, consequential, or
        punitive damages, or for any loss of data, profits, revenue, goodwill, or
        business, arising out of or in connection with your use of or inability
        to use the Demo, including any reliance on AI-generated output or any
        charges incurred on your own provider API keys.
      </p>
      <p className="uppercase">
        To the extent liability cannot be excluded, it is limited in aggregate to
        the greater of the amount you paid to use the Demo (which is zero) or EUR
        50.
      </p>
      <p>
        Nothing in these terms excludes liability that cannot lawfully be
        excluded, such as liability for death or personal injury caused by
        negligence, or for fraud. Some jurisdictions do not allow certain
        exclusions, so parts of this section may not apply to you.
      </p>

      <h2>11. Indemnity</h2>
      <p>
        You agree to indemnify and hold harmless the operator of the Demo against
        claims, damages, and costs (including reasonable legal fees) arising from
        content you submit or from your breach of these terms or of applicable
        law.
      </p>

      <h2>12. Termination</h2>
      <p>
        You may stop using the Demo and delete your account at any time from
        Settings. Access may be suspended or terminated at any time, for any
        reason, including shutting the project down entirely. Sections 5 through
        11 survive termination.
      </p>

      <h2>13. Open source</h2>
      <p>
        The source code is published under the MIT License, which governs the
        code itself. These terms govern use of this hosted instance. The MIT
        License&rsquo;s own warranty disclaimer applies to the code.
      </p>

      <h2>14. Changes</h2>
      <p>
        These terms may change; the date at the top reflects the latest version.
        Continuing to use the Demo after a change means you accept it.
      </p>

      <h2>15. Governing law</h2>
      <p>
        These terms are governed by the laws of the operator&rsquo;s place of
        residence, without regard to conflict-of-law rules. If you are a
        consumer, you keep the protection of any mandatory provisions of the law
        of your own country of residence.
      </p>

      <h2>16. Contact</h2>
      <p>
        Questions: <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>. See
        also the <Link href="/privacy">Privacy Policy</Link>.
      </p>
    </article>
  )
}
