
"use client";

import { Fight } from "@/types";
import { Trophy } from "lucide-react";
import FighterModal from "./FighterModal";
import Image from "next/image";

interface FightCardProps {
    fight: Fight;
    oddsFormat: 'percent' | 'american';
}

export default function FightCard({ fight, oddsFormat }: FightCardProps) {
    const winner = fight.prediction?.winner;
    const isF1Winner = winner === fight.fighter_1;

    // Odds Conversion Helper
    const formatOdds = (probStr: string) => {
        if (oddsFormat === 'percent') return probStr;

        const p = parseFloat(probStr.replace('%', '')) / 100;
        if (p >= 0.999) return "-10000+";
        if (p <= 0.001) return "+10000+";

        if (p > 0.5) {
            const american = - (p / (1 - p)) * 100;
            return Math.round(american).toString();
        } else {
            const american = ((1 - p) / p) * 100;
            return "+" + Math.round(american).toString();
        }
    };


    // Get win probabilities
    const f1Prob = fight.prediction?.odds?.[fight.fighter_1] || "50%";
    const f2Prob = fight.prediction?.odds?.[fight.fighter_2] || "50%";
    const f1Percent = parseFloat(f1Prob.replace('%', ''));
    const f2Percent = parseFloat(f2Prob.replace('%', ''));

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

    // Dynamic styles based on Title Fight status
    const isTitle = fight.is_title_fight;
    const containerClasses = isTitle
        ? "bg-zinc-900/80 border border-zinc-800 border-b-4 border-b-[#BF953F] shadow-[0_4px_20px_rgba(191,149,63,0.1)]"
        : "bg-zinc-900/50 hover:bg-zinc-900 border border-zinc-800 hover:border-red-900/50";

    const winnerBorderColor = "border-green-400"; // Soft Green
    const winnerTextColor = "text-green-400";

    return (
        <FighterModal fight={fight} oddsFormat={oddsFormat}>
            <div className={`group relative transition-all duration-300 rounded-lg p-4 cursor-pointer overflow-hidden ${containerClasses}`}>

                {/* Hover Effect Border Gradient */}
                {!isTitle && (
                    <div className="absolute inset-0 bg-linear-to-r from-transparent via-red-900/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 pointer-events-none" />
                )}

                <div className="flex justify-between items-center relative z-10 gap-1 sm:gap-4">
                    {/* Fighter 1 */}
                    <div className="flex-1 flex items-center gap-2 sm:gap-3 min-w-0">
                        {f1Image && (
                            <div className={`relative w-10 h-10 sm:w-14 sm:h-14 shrink-0 rounded-full overflow-hidden border-2 ${isF1Winner ? winnerBorderColor : 'border-zinc-700'}`}>
                                <img
                                    src={f1Image}
                                    alt={fight.fighter_1}
                                    className="w-full h-full object-cover"
                                />
                            </div>
                        )}
                        <div className="flex flex-col items-start min-w-0">
                            <span className={`text-sm sm:text-lg font-bold truncate w-full ${isF1Winner ? 'text-white' : 'text-zinc-400'} ${isF1Winner ? 'drop-shadow-[0_0_8px_rgba(74,222,128,0.5)]' : ''}`}>
                                {fight.fighter_1}
                            </span>
                            <span className={`text-[10px] sm:text-xs font-mono mt-0.5 ${isF1Winner ? winnerTextColor : 'text-zinc-500'}`}>
                                {formatOdds(f1Prob)}
                            </span>
                        </div>
                    </div>

                    {/* VS Badge */}
                    <div className="px-1 sm:px-3 shrink-0">
                        <span className={`font-black italic text-base sm:text-xl ${isTitle ? 'text-[#BF953F]/40' : 'text-zinc-800'}`}>VS</span>
                    </div>

                    {/* Fighter 2 */}
                    <div className="flex-1 flex items-center justify-end gap-2 sm:gap-3 min-w-0">
                        <div className="flex flex-col items-end text-right min-w-0">
                            <span className={`text-sm sm:text-lg font-bold truncate w-full ${!isF1Winner && winner ? 'text-white' : 'text-zinc-400'} ${!isF1Winner && winner ? 'drop-shadow-[0_0_8px_rgba(74,222,128,0.5)]' : ''}`}>
                                {fight.fighter_2}
                            </span>
                            <span className={`text-[10px] sm:text-xs font-mono mt-0.5 ${!isF1Winner && winner ? winnerTextColor : 'text-zinc-500'}`}>
                                {formatOdds(f2Prob)}
                            </span>
                        </div>
                        {f2Image && (
                            <div className={`relative w-10 h-10 sm:w-14 sm:h-14 shrink-0 rounded-full overflow-hidden border-2 ${!isF1Winner && winner ? winnerBorderColor : 'border-zinc-700'}`}>
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

                {/* Bottom Stats: Market Odds & Winner Badge */}
                <div className="mt-3 flex items-center justify-between relative z-10">
                    {/* Market Odds F1 */}
                    <div className="flex-1">
                        {fight.market_odds?.[fight.fighter_1] && (
                            <div className="flex flex-col items-start translate-y-2">
                                <span className="text-[8px] uppercase tracking-widest font-bold text-zinc-500">Market</span>
                                <span className="text-xs sm:text-sm font-mono font-bold text-white">
                                    {Number(fight.market_odds[fight.fighter_1]) > 0 ? `+${fight.market_odds[fight.fighter_1]}` : fight.market_odds[fight.fighter_1]}
                                </span>
                            </div>
                        )}
                    </div>

                    {/* Prediction Badge */}
                    {winner && (
                        <div className="flex justify-center px-2">
                            <span className="text-xs font-bold uppercase tracking-wider flex items-center gap-1 bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full border border-zinc-800">
                                <Trophy size={12} className={isTitle ? "text-[#BF953F]" : "text-yellow-500"} />
                                <span className="text-white">{winner}</span>
                                <span className="text-zinc-500">-</span>
                                <span className={winnerTextColor}>{formatOdds(fight.prediction?.confidence || "0%")}</span>
                            </span>
                        </div>
                    )}

                    {/* Market Odds F2 */}
                    <div className="flex-1 text-right">
                        {fight.market_odds?.[fight.fighter_2] && (
                            <div className="flex flex-col items-end translate-y-2">
                                <span className="text-[8px] uppercase tracking-widest font-bold text-zinc-500">Market</span>
                                <span className="text-xs sm:text-sm font-mono font-bold text-white">
                                    {Number(fight.market_odds[fight.fighter_2]) > 0 ? `+${fight.market_odds[fight.fighter_2]}` : fight.market_odds[fight.fighter_2]}
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </FighterModal>
    );
}
