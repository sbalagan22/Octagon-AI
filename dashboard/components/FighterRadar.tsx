
"use client";

import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Legend } from "recharts";
import { ChartData } from "@/types";

interface FighterRadarProps {
    data: ChartData;
    fighter1Name: string;
    fighter2Name: string;
}

export default function FighterRadar({ data, fighter1Name, fighter2Name }: FighterRadarProps) {
    // Transform data for Recharts
    // Recharts radar expects array of objects: { subject: 'Age', A: 120, B: 110, fullMark: 150 }
    const chartData = data.labels.map((label, index) => ({
        subject: label,
        A: data.fighter_1_data[index] * 100, // Convert 0.5 -> 50 for better scale visuals? Or keep 0-1
        B: data.fighter_2_data[index] * 100,
        fullMark: 100
    }));

    return (
        <div className="w-full h-[300px] md:h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                    <PolarGrid stroke="#444" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#888', fontSize: 12 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar
                        name={fighter1Name}
                        dataKey="A"
                        stroke="#D20A0A"
                        strokeWidth={3}
                        fill="#D20A0A"
                        fillOpacity={0.4}
                    />
                    <Radar
                        name={fighter2Name}
                        dataKey="B"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        fill="#3b82f6"
                        fillOpacity={0.4}
                    />
                    <Legend wrapperStyle={{ color: '#fff' }} />
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
}
