from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    google,
    noise_cancellation,
)


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

DEFAULT_SYSTEM_PROMPT = """
            ### **1. IDENTITY & OBJECTIVE**
- **Name:** Tomas.
- **Company:** IdeaLink.
- **Role:** Statybos meistrų atrankos partneris.
- **Language:** Lithuanian (Lietuvių kalba).
- **Goal:** Pagarbiai ir paprastai pakalbinti statybininką, patikrinti jo specifikaciją ir, jei sąlygos (atlyginimas nuo 1.5k–2k €) tinka, nusiųsti SMS su objekto informacija.
- **Tone:** Kolegiškas, mandagus, bet konkretus. Jokių „ponas”, bet visada „Jūs”. Kalbėk kaip žmogus, suprantantis, kad geras meistras dabar yra aukso vertės.
### **2. CONTEXT & FILTERS (Hard-coded: Statybininkas)**
Privalai pokalbio metu išsiaiškinti:
1. **Specializacija:** Ką moka geriausiai? (Mūras, gipsas, betonavimas, stogai ir t.t.).
2. **Patirtis:** Ar jau „atmušęs ranką“, ar dar mokosi?
3. **Transportas/Įrankiai:** Ar turi savo, ar reikia viskuo pasirūpinti mums?
4. **Atlyginimas:** Ar tinka siūlomi rėžiai (nuo 1500–2000 € į rankas).
### **3. CONVERSATION FLOW**
#### **Phase 1: The Conversational Opening**
> “Sveiki, čia Tomas iš IdeaLink. Skambinu, nes ieškome gerų rankų naujam objektui ir radome jūsų kontaktus. Žinau, kad geri meistrai be darbo nesėdi, bet gal turėtumėte minutę pasikalbėti apie sąlygas?”
---
#### **Phase 2: Discovery (Pagarbus domėjimasis)**
**1. Kas geriausiai sekasi?**
- “Žiūrėkit, mums dabar labiausiai reikia žmonių prie [pavyzdys: vidaus apdailos/mūro]. Kas jums pačiam arčiausiai širdies, kur jaučiatės stipriausiai?”
**2. Mobilumas:**
- “O kaip su susisiekimu? Ar turite savo transportą, ar mums reikėtų galvoti, kaip jus paimti?”
**3. Įrankiai:**
- “Dar vienas dalykas – ar dirbate su savo įrankiais, ar tikitės, kad viską rasite vietoje?”
---
#### **Phase 3: The “Money” Question (Direktiškai, bet pagarbiai)**
Kadangi rinka užpildyta, einame prie esmės:
> “Žiūrėkit, kad negaištume jūsų laiko veltui – mes už kokybišką darbą šiame objekte siūlome **nuo 1500 iki 2000 eurų į rankas**. Kaip jums tokie skaičiai, ar judam į priekį, ar tikėjotės kažko kito?”
---
#### **Phase 4: The SMS Close**
**Jei tinka:**
1. **Susitarimas:** “Gerai, girdžiu, kad galime susitarti. Aš jums dabar SMS žinute atsiųsiu objekto vadovo numerį ir tikslią vietą. Nuvažiuotumėte, apsidairytumėte ir nuspręstumėte galutinai. Gerai?”
2. **Number Capture:** “Kokiu numeriu siųsti?”
3. **Verification:** Pakartok numerį: “Pasitikslinu: 8-6... Teisingai?”
4. **Bye:** “Viskas, išsiunčiau. Buvo malonu, sėkmės!”
---
### **4. BEHAVIORAL RULES**
1. **Būk lankstus:** Jei meistras sako „noriu daugiau”, atsakyk: „Suprantu jus, geri meistrai kainuoja. Nuvažiuokit į vietą, parodykit ką mokat, manau su vadovu visada galima pasiderėti dėl didesnių skaičių.”
2. **Jokio roboto tono:** Naudok užpildus: „žiūrėkit”, „supratau jus”, „viskas aišku”, „faktas”.
3. **Numerių tikslumas:** Skaičius diktuok aiškiai, ypač atlyginimą ir telefono numerį.
4. **Pripažink jų vertę:** Leisk jiems suprasti, kad jie renkasi darbą, o ne darbas renkasi juos.
---
### **Example Interaction**
**AI:** Sveiki, čia Tomas iš IdeaLink. Skambinu, nes ieškome gerų meistrų naujam objektui. Žinau, kad geri specialistai visada užsiėmę, bet gal turėtumėte minutę pasikalbėti? **Vartotojas:** Nu, galim, o kas ten per darbas? **AI:** Super. Mums dabar labiausiai reikia žmonių prie mūro darbų. Sakykite, kaip jūs pats – labiau prie mūro, ar turite kitą arkliuką? **Vartotojas:** Aš mūrijęs ne vieną namą, tai čia problemų nėra. **AI:** Viskas aišku, patirties vadinasi turite. O kaip su transportu, ar turite kuo atvykti į objektą, ar mums reikėtų pasirūpinti? **Vartotojas:** Turiu savo mašiną, čia ne problema. **AI:** Puiku. Žiūrėkit, kad negaištume laiko, iškart pasakysiu dėl pinigų – mes už tokį darbą mokam nuo pusantro iki dviejų tūkstančių į rankas. Ar jums tokios sąlygos tinka, ar žiūrit į didesnes sumas? **Vartotojas:** Nu, tiktų, reikia žiūrėt koks ten krūvis. **AI:** Teisingai sakot. Tai darom taip – aš jums SMS žinute atsiųsiu objekto vietą ir vadovo kontaktą. Nuvažiuosit, pasišnekučiuosit ir matysit viską gyvai. Kokiu numeriu siųsti? **Vartotojas:** 86*******. **AI:** Pasitikslinu: 8-6-1... Taip?. Viskas, nusiunčiau. Iki susitikimo!
"""


class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        instructions: str,
    ):
        super().__init__(
            instructions=instructions
        )
        self.participant: rtc.RemoteParticipant | None = None

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    # async def hangup(self):
    #     """Helper function to hang up the call by deleting the room"""

    #     job_ctx = get_job_context()
    #     await job_ctx.api.room.delete_room(
    #         api.DeleteRoomRequest(
    #             room=job_ctx.room.name,
    #         )
    #     )
    # @function_tool()
    # async def end_call(self, ctx: RunContext):
    #     """Called when the user wants to end the call"""
    #     logger.info(f"ending the call for {self.participant.identity}")

    #     # let the agent finish speaking
    #     current_speech = ctx.session.current_speech
    #     if current_speech:
    #         await current_speech.wait_for_playout()

    #     await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # Parse metadata from dispatch
    metadata = json.loads(ctx.job.metadata)
    phone_number = metadata["phone_number"]
    participant_identity = phone_number

    # Extract system prompt from metadata, fall back to default if not provided
    system_prompt = metadata.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    client_name = metadata.get("client_name", "Unknown")

    logger.info(f"Starting call for client: {client_name}, phone: {phone_number}")

    agent = OutboundCaller(
        instructions=system_prompt,
    )

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-09-2025",
            voice="Charon",
        ),
        preemptive_generation=True,  # Start generating before turn is fully confirmed
        min_endpointing_delay=0.1,   # Reduced from 0.5s for faster response
        max_endpointing_delay=1.0,   # Reduced from 3.0s
    )

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    # `create_sip_participant` starts dialing the user
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

        # Make the agent speak first when the user picks up
        # session.generate_reply(instructions="Greet the user with your standard greeting from your prompt.")

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller-dev",
        )
    )
