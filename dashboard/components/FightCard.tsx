
"use client";

import { Fight } from "@/types";
import { Trophy } from "lucide-react";
import FighterModal from "./FighterModal";
import Image from "next/image";

interface FightCardProps {
    fight: Fight;
}

export default function FightCard({ fight }: FightCardProps) {
    const winner = fight.prediction?.winner;
    const isF1Winner = winner === fight.fighter_1;

    // Get win percentages from odds
    const f1Odds = fight.prediction?.odds?.[fight.fighter_1] || "50%";
    const f2Odds = fight.prediction?.odds?.[fight.fighter_2] || "50%";
    const f1Percent = parseFloat(f1Odds.replace('%', ''));
    const f2Percent = parseFloat(f2Odds.replace('%', ''));

    // Generate ESPN image URLs from fighter URLs
    const getImageUrl = (espnUrl: string) => {
        const match = espnUrl.match(/\/id\/(\d+)\//);
        if (match) {
            return `https://a.espncdn.com/combiner/i?img=/i/headshots/mma/players/full/${match[1]}.png&w=350&h=254`;
        }
        return null;
    };

    const f1Image = getImageUrl(fight.fighter_1_url);
    const f2Image = getImageUrl(fight.fighter_2_url);

    return (
        <FighterModal fight={fight}>
            <div className="group relative bg-zinc-900/50 hover:bg-zinc-900 border border-zinc-800 hover:border-red-900/50 transition-all duration-300 rounded-lg p-4 cursor-pointer overflow-hidden">

                {/* Hover Effect Border Gradient */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-red-900/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 pointer-events-none" />

                <div className="flex justify-between items-center relative z-10 gap-1 sm:gap-4">
                    {/* Fighter 1 */}
                    <div className="flex-1 flex items-center gap-2 sm:gap-3 min-w-0">
                        {f1Image && (
                            <div className={`relative w-10 h-10 sm:w-14 sm:h-14 shrink-0 rounded-full overflow-hidden border-2 ${isF1Winner ? 'border-red-500' : 'border-zinc-700'}`}>
                                <img
                                    src={f1Image}
                                    alt={fight.fighter_1}
                                    className="w-full h-full object-cover"
                                />
                            </div>
                        )}
                        <div className="flex flex-col items-start min-w-0">
                            <span className={`text-sm sm:text-lg font-bold truncate w-full ${isF1Winner ? 'text-red-500' : 'text-zinc-300'} ${isF1Winner ? 'drop-shadow-[0_0_8px_rgba(220,38,38,0.5)]' : ''}`}>
                                {fight.fighter_1}
                            </span>
                            <span className={`text-[10px] sm:text-xs font-mono mt-0.5 ${isF1Winner ? 'text-red-500' : 'text-zinc-500'}`}>
                                {f1Odds}
                            </span>
                        </div>
                    </div>

                    {/* VS Badge */}
                    <div className="px-1 sm:px-3 shrink-0">
                        <span className="text-zinc-800 font-black italic text-base sm:text-xl">VS</span>
                    </div>

                    {/* Fighter 2 */}
                    <div className="flex-1 flex items-center justify-end gap-2 sm:gap-3 min-w-0">
                        <div className="flex flex-col items-end text-right min-w-0">
                            <span className={`text-sm sm:text-lg font-bold truncate w-full ${!isF1Winner && winner ? 'text-blue-500' : 'text-zinc-300'} ${!isF1Winner && winner ? 'drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]' : ''}`}>
                                {fight.fighter_2}
                            </span>
                            <span className={`text-[10px] sm:text-xs font-mono mt-0.5 ${!isF1Winner && winner ? 'text-blue-500' : 'text-zinc-500'}`}>
                                {f2Odds}
                            </span>
                        </div>
                        {f2Image && (
                            <div className={`relative w-10 h-10 sm:w-14 sm:h-14 shrink-0 rounded-full overflow-hidden border-2 ${!isF1Winner && winner ? 'border-blue-500' : 'border-zinc-700'}`}>
                                <img
                                    src={f2Image}
                                    alt={fight.fighter_2}
                                    className="w-full h-full object-cover"
                                />
                            </div>
                        )}
                    </div>
                </div>

                {/* Prediction Bar */}
                {winner && (
                    <div className="mt-4 w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden flex">
                        <div
                            className="h-full bg-red-600 transition-all duration-500"
                            style={{ width: `${f1Percent}%` }}
                        />
                        <div
                            className="h-full bg-blue-500 transition-all duration-500"
                            style={{ width: `${f2Percent}%` }}
                        />
                    </div>
                )}

                {/* Winner Badge */}
                {winner && (
                    <div className="mt-2 flex justify-center">
                        <span className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1 ${isF1Winner ? 'text-red-500' : 'text-blue-500'}`}>
                            <Trophy size={12} />
                            {winner} - {fight.prediction?.confidence}
                        </span>
                    </div>
                )}
            </div>
        </FighterModal>
    );
}
