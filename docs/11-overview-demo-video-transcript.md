# Spotter Demo Video Transcript

## Demo Concept

`Spotter: The gym coach for people who would rather not be judged`

## Characters

- `Linh`: wants to work out, but avoids the gym
- `Minh`: friend, funny but supportive
- Optional `Narrator`: can be one of the two, or a voiceover

## Tone

- Light, relatable, slightly comedic
- Short lines, each line reveals the pain point
- Then pivot into a polished product demo

## Total Length

- Around 2.5 to 3 minutes

## Script

### Scene 1: The Problem

`0:00 - 0:20`

Visual:
Linh stands in workout clothes at home, staring at a gym bag like it personally offended her.

Minh walks in.

Minh:
"You're not going?"

Linh:
"It's too far."

Beat.

Minh:
"That's not the real reason."

Linh:
"Fine. I don't want to be the girl doing squats wrong in front of twenty strangers and one guy filming his triceps."

Minh:
"Fair."

Linh:
"And a private trainer costs more than my groceries."

Small beat.

Minh:
"So your fitness plan is... emotional support and denial?"

Linh looks at the gym bag.

Linh:
"Currently, yes."

Why this works:

- `too far` = distance
- `doing squats wrong in front of strangers` = shame / fear of judgment
- `private trainer costs more than my groceries` = PT too expensive
- funny but very clear target customer

### Scene 2: The Reframe

`0:20 - 0:35`

Visual:
Minh picks up Linh's phone.

Minh:
"What if you could train at home... and still get real feedback?"

Linh:
"From who? The front camera?"

Minh:
"From AI. But not the kind that just says 'great job' and disappears."

Cut to product UI.

Voiceover:
"That's exactly why we built Spotter."

### Scene 3: Product Intro

`0:35 - 0:55`

Visual:
Screen recording of Spotter homepage and upload flow.

Voiceover:
"Spotter is an AI workout coach for people who want to train seriously, but don't want the cost, distance, or pressure of the gym. You upload a workout video, and Spotter turns it into a structured, explainable coaching report."

Visual:
Show progress steps:

- video quality
- pose tracking
- exercise classification
- rep counting
- issue detection
- coach summary

Voiceover:
"It doesn't just watch a video and guess. It checks video quality, tracks pose, identifies the exercise, counts reps, analyzes each rep, detects valid variations, flags real issues, and then generates a grounded coach summary."

### Scene 4: The Magic

`0:55 - 1:30`

Visual:
Show real squat run in the app.
Open annotated video, issue clips, summary tab.

Voiceover:
"Here's the key difference: Spotter doesn't give generic advice like 'keep your chest up' and call it a day. It shows what exercise you did, how many reps you completed, which reps changed, what variation was valid, and what issue actually appeared."

Visual:
Highlight:

- `squat`
- rep count
- `wide_squat_stance`
- `shallow_depth`

Voiceover:
"In this example, Spotter recognizes a wide squat stance as a valid variation, not a mistake. But it still catches shallow depth on specific reps. So instead of overcorrecting everything, it separates style from actual form breakdown."

### Scene 5: Why It's Trustworthy

`1:30 - 1:55`

Visual:
Show JSON tab or artifacts view briefly.
Show Coach tab with provider/model/source.

Voiceover:
"And we built guardrails into the system. The coach summary is generated from structured artifacts, not from vague intuition. Then a verifier checks whether the model mentioned issues outside the evidence, overcorrected a valid variation, or drifted into unsafe claims."

Visual:
Quick highlight:

- evidence artifacts
- verifier
- fallback logic

Voiceover:
"If the model gets too creative, Spotter falls back to a safer summary. So the user gets feedback they can actually trust."

### Scene 6: Sponsor Tech

`1:55 - 2:15`

Visual:
Subtle overlays or callouts:

- Hugging Face Spaces
- ZeroGPU
- local Nemotron coach model
- Hugging Face Hub

Voiceover:
"We also leaned into the Hugging Face ecosystem to make this real. Spotter runs beautifully in Hugging Face Spaces, uses ZeroGPU for demo-ready compute, runs a local Nemotron coach model, and loads our exercise router through Hugging Face Hub."

### Scene 7: Return To Characters

`2:15 - 2:40`

Visual:
Back to Linh and Minh at home.
Linh finishes a set, checks the app, sees feedback.

Minh:
"So... no gym?"

Linh:
"No commute. No crowd. No fake confidence."

Minh:
"And no trainer bill?"

Linh:
"Exactly."

Beat.

Minh:
"So now the only thing judging you... is a grounded, evidence-based AI."

Linh:
"Which is somehow less terrifying than people."

They both laugh.

### Scene 8: Closing Line

`2:40 - 2:55`

Visual:
Product hero shot, annotated video, summary, issue clips.

Voiceover:
"Spotter is for the people who want to get stronger, but not embarrassed. For the people who want real coaching, but not gym anxiety. For the people who don't need hype. They need feedback."

Final line on screen:
`Spotter: Your coach, without the crowd.`

## Why This Version Is Strong

- The opening quickly locks onto the right customer:
  - the gym is far away
  - they feel uncomfortable in crowded spaces
  - they are afraid of being judged for bad form
  - one-on-one PT is too expensive
- The dialogue is short, punchy, and easy to perform
- It has a little humor without becoming silly
- It transitions naturally into the product demo
- It makes room to highlight sponsor tech
- It ends on user emotion, not just a feature list

## Optional Funnier Variations

You can replace:

Minh:
"So your fitness plan is... emotional support and denial?"

with:

Minh:
"So your current workout split is... anxiety, excuses, and one YouTube warm-up?"

Or replace the ending with:

Minh:
"So now the only thing judging you is AI."

Linh:
"Perfect. AI doesn't smirk."
