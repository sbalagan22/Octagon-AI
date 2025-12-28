
"use client";

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from "@/components/ui/dialog";
import { Fight } from "@/types";
import { User, Scale, Ruler, Activity, Trophy, Target, Clock, X } from "lucide-react";
import Image from "next/image";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, Legend } from "recharts";

interface FighterModalProps {
    fight: Fight;
    children: React.ReactNode;
}

export default function FighterModal({ fight, children }: FighterModalProps) {
    const getFighterImage = (url: string, name: string) => {
        const idMatch = url.match(/\/id\/(\d+)\//);
        if (idMatch) {
            return `https://a.espncdn.com/combiner/i?img=/i/headshots/mma/players/full/${idMatch[1]}.png&w=350&h=254`;
        }
        return "https://placehold.co/400x600/D20A0A/white?text=" + name.split(" ")[0];
    };

    const f1Image = getFighterImage(fight.fighter_1_url, fight.fighter_1);
    const f2Image = getFighterImage(fight.fighter_2_url, fight.fighter_2);

    const factors = fight.prediction?.factors;
    const f1Stats = factors?.[fight.fighter_1];
    const f2Stats = factors?.[fight.fighter_2];

    // Generate radar chart data from prediction factors
    const generateRadarData = () => {
        if (!f1Stats || !f2Stats) return null;

        // Normalize stats to 0-100 scale for radar
        const maxSlpm = Math.max(f1Stats.slpm, f2Stats.slpm, 5);
        const maxTd = Math.max(f1Stats.td_rate, f2Stats.td_rate, 3);
        const maxCtrl = Math.max(f1Stats.ctrl_rate, f2Stats.ctrl_rate, 200);
        const maxKd = Math.max(f1Stats.kd_rate, f2Stats.kd_rate, 1);
        const maxSub = Math.max(f1Stats.sub_rate, f2Stats.sub_rate, 2);

        return [
            {
                stat: 'Striking',
                [fight.fighter_1]: (f1Stats.slpm / maxSlpm) * 100,
                [fight.fighter_2]: (f2Stats.slpm / maxSlpm) * 100
            },
            {
                stat: 'Takedowns',
                [fight.fighter_1]: (f1Stats.td_rate / maxTd) * 100,
                [fight.fighter_2]: (f2Stats.td_rate / maxTd) * 100
            },
            {
                stat: 'Control',
                [fight.fighter_1]: (f1Stats.ctrl_rate / maxCtrl) * 100,
                [fight.fighter_2]: (f2Stats.ctrl_rate / maxCtrl) * 100
            },
            {
                stat: 'KO Power',
                [fight.fighter_1]: (f1Stats.kd_rate / maxKd) * 100,
                [fight.fighter_2]: (f2Stats.kd_rate / maxKd) * 100
            },
            {
                stat: 'Submissions',
                [fight.fighter_1]: (f1Stats.sub_rate / maxSub) * 100,
                [fight.fighter_2]: (f2Stats.sub_rate / maxSub) * 100
            },
            {
                stat: 'Win Rate',
                [fight.fighter_1]: f1Stats.win_rate,
                [fight.fighter_2]: f2Stats.win_rate
            }
        ];
    };

    const radarData = generateRadarData();

    const StatRow = ({ label, value, icon: Icon }: { label: string; value: string | number; icon?: any }) => (
        <div className="flex justify-between items-center py-2 border-b border-zinc-800/50">
            <span className="text-zinc-500 text-[10px] sm:text-xs flex items-center gap-1.5 uppercase font-medium">
                {Icon && <Icon size={12} className="text-zinc-400" />}
                {label}
            </span>
            <span className="font-bold text-white text-xs sm:text-sm">{value}</span>
        </div>
    );

    return (
        <Dialog>
            <DialogTrigger asChild>
                {children}
            </DialogTrigger>
            <DialogContent className="w-full sm:max-w-5xl bg-zinc-950 border-zinc-800 text-white p-0 overflow-hidden h-dvh sm:h-auto sm:max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <DialogHeader className="p-3 sm:p-4 bg-linear-to-r from-red-950/50 via-zinc-900 to-blue-950/50 border-b border-zinc-800 sticky top-0 z-50 backdrop-blur-md flex flex-row items-center justify-between">
                    <DialogTitle className="sr-only">Fighter Details</DialogTitle>
                    <div className="flex-1 text-center">
                        {/* Title removed as requested */}
                    </div>
                    <DialogClose className="p-2 hover:bg-white/10 rounded-full transition-colors">
                        <X size={20} className="text-zinc-400 group-hover:text-white" />
                    </DialogClose>
                </DialogHeader>

                <div className="grid grid-cols-1 md:grid-cols-3 pb-safe">
                    {/* Fighter 1 (Red) */}
                    <div className="bg-linear-to-br from-red-950/20 to-zinc-950 p-4 flex flex-col items-center">
                        <div className="relative w-24 h-24 md:w-32 md:h-32 mb-3 rounded-full overflow-hidden border-3 border-red-600 shadow-[0_0_15px_rgba(220,38,38,0.3)]">
                            <Image src={f1Image} alt={fight.fighter_1} fill className="object-cover" />
                        </div>
                        <h3 className="text-lg font-black uppercase text-center mb-1">{fight.fighter_1}</h3>
                        <span className="text-red-500 font-bold text-xs tracking-widest mb-3">RED CORNER</span>

                        {/* Win Probability */}
                        <div className="text-center mb-3">
                            <span className="text-2xl font-black text-red-500">
                                {fight.prediction?.odds?.[fight.fighter_1] || '50%'}
                            </span>
                            <div className="text-xs text-zinc-500 uppercase">Win Probability</div>
                        </div>

                        {/* Stats */}
                        {f1Stats && (
                            <div className="w-full space-y-0.5 text-sm bg-zinc-900/50 rounded-lg p-3">
                                <StatRow label="Record" value={`${f1Stats.wins}-${f1Stats.losses}`} icon={Trophy} />
                                <StatRow label="Recent Form" value={f1Stats.recent_form && f1Stats.recent_form !== 'N/A' ? f1Stats.recent_form : 'Unknown'} icon={Activity} />
                                <StatRow label="Height" value={`${Math.round(f1Stats.height_cm)} cm`} icon={Ruler} />
                                <StatRow label="Reach" value={`${Math.round(f1Stats.reach_cm)} cm`} icon={Target} />
                            </div>
                        )}
                    </div>

                    {/* Center (Radar Chart) */}
                    <div className="p-4 sm:p-6 flex flex-col justify-center items-center border-x border-zinc-900 bg-zinc-950/50">
                        {radarData ? (
                            <div className="w-full h-[250px] sm:h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="65%" data={radarData}>
                                        <PolarGrid stroke="#333" />
                                        <PolarAngleAxis dataKey="stat" tick={{ fill: '#888', fontSize: 9 }} />
                                        <Radar
                                            name={fight.fighter_1}
                                            dataKey={fight.fighter_1}
                                            stroke="#D20A0A"
                                            strokeWidth={2}
                                            fill="#D20A0A"
                                            fillOpacity={0.3}
                                        />
                                        <Radar
                                            name={fight.fighter_2}
                                            dataKey={fight.fighter_2}
                                            stroke="#3b82f6"
                                            strokeWidth={2}
                                            fill="#3b82f6"
                                            fillOpacity={0.3}
                                        />
                                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>
                        ) : (
                            <div className="text-zinc-500 text-center text-sm">No statistical data available</div>
                        )}

                        {/* Prediction Result */}
                        <div className="mt-3 text-center">
                            <div className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Prediction</div>
                            <div className="text-lg font-bold">
                                {fight.prediction?.winner ? (
                                    <span className={fight.prediction.winner === fight.fighter_1 ? "text-red-500" : "text-blue-500"}>
                                        {fight.prediction.winner}
                                    </span>
                                ) : "Undecided"}
                            </div>
                            <div className="text-xl font-black text-green-500">
                                {fight.prediction?.confidence}
                            </div>
                        </div>
                    </div>

                    {/* Fighter 2 (Blue) */}
                    <div className="bg-linear-to-bl from-blue-950/20 to-zinc-950 p-4 flex flex-col items-center">
                        <div className="relative w-24 h-24 md:w-32 md:h-32 mb-3 rounded-full overflow-hidden border-3 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
                            <Image src={f2Image} alt={fight.fighter_2} fill className="object-cover" />
                        </div>
                        <h3 className="text-lg font-black uppercase text-center mb-1">{fight.fighter_2}</h3>
                        <span className="text-blue-500 font-bold text-xs tracking-widest mb-3">BLUE CORNER</span>

                        {/* Win Probability */}
                        <div className="text-center mb-3">
                            <span className="text-2xl font-black text-blue-500">
                                {fight.prediction?.odds?.[fight.fighter_2] || '50%'}
                            </span>
                            <div className="text-xs text-zinc-500 uppercase">Win Probability</div>
                        </div>

                        {/* Stats */}
                        {f2Stats && (
                            <div className="w-full space-y-0.5 text-sm bg-zinc-900/50 rounded-lg p-3">
                                <StatRow label="Record" value={`${f2Stats.wins}-${f2Stats.losses}`} icon={Trophy} />
                                <StatRow label="Recent Form" value={f2Stats.recent_form && f2Stats.recent_form !== 'N/A' ? f2Stats.recent_form : 'Unknown'} icon={Activity} />
                                <StatRow label="Height" value={`${Math.round(f2Stats.height_cm)} cm`} icon={Ruler} />
                                <StatRow label="Reach" value={`${Math.round(f2Stats.reach_cm)} cm`} icon={Target} />
                            </div>
                        )}
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}
