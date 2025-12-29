
"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Calendar, MapPin, ChevronRight, Trophy } from "lucide-react";
import FightCard from "@/components/FightCard";
import { Event } from "@/types";

export default function Home() {
    const [events, setEvents] = useState<Event[]>([]);
    const [featuredEvent, setFeaturedEvent] = useState<Event | null>(null);
    const [oddsFormat, setOddsFormat] = useState<'percent' | 'american'>('percent');

    useEffect(() => {
        fetch("/upcoming_events.json")
            .then((res) => res.json())
            .then((data: Event[]) => {
                setEvents(data);
                if (data.length > 0) {
                    setFeaturedEvent(data[0]);
                }
            })
            .catch((err) => console.error("Failed to load events", err));
    }, []);

    if (!featuredEvent) {
        return (
            <div className="flex items-center justify-center h-screen bg-black text-red-600 animate-pulse font-black italic tracking-widest">
                OCTAGON AI...
            </div>
        );
    }

    // Get all events for the sidebar
    const upcomingEvents = events;

    return (
        <main className="flex flex-col lg:flex-row min-h-screen bg-black">
            {/* LEFT: Featured Event (Takes up 2/3 on desktop) */}
            <div className="flex-1 lg:flex-2 overflow-y-auto border-r border-zinc-900 scrollbar-thin scrollbar-thumb-zinc-800">

                {/* HERO SECTION */}
                <div className="relative min-h-[220px] sm:min-h-[300px] w-full flex items-end p-4 sm:p-6 overflow-hidden bg-zinc-900 border-b border-zinc-800">
                    {/* Background Gradient/Image Placeholder */}
                    <div className="absolute inset-0 bg-[url('https://placehold.co/1920x1080/1a1a1a/333333?text=UFC+Event+BG')] bg-cover bg-center opacity-30 mix-blend-overlay"></div>
                    <div className="absolute inset-0 bg-linear-to-t from-black via-black/80 to-transparent"></div>

                    {/* Content */}
                    <div className="relative z-10 w-full max-w-5xl mx-auto">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 mb-8">
                            <div className="flex items-center gap-4">
                                <div className="relative w-14 h-14 sm:w-16 sm:h-16 overflow-hidden rounded-xl border border-white/10 shadow-2xl">
                                    <Image src="/logo.png" alt="Octagon AI Logo" fill className="object-cover scale-110" />
                                </div>
                                <div className="flex flex-col">
                                    <h1 className="text-2xl sm:text-4xl font-black italic tracking-tighter text-white leading-none">
                                        OCTAGON <span className="text-red-600">AI</span>
                                    </h1>
                                    <span className="text-xs uppercase tracking-[0.3em] font-bold text-zinc-500 mt-1">Predictive Intelligence</span>
                                </div>
                            </div>

                            {/* Odds Toggle */}
                            <div className="flex items-center p-1 bg-zinc-900/80 rounded-lg border border-zinc-800 self-start sm:self-center">
                                <button
                                    onClick={() => setOddsFormat('percent')}
                                    className={`px-3 py-1 text-[10px] font-bold uppercase tracking-widest transition-all rounded ${oddsFormat === 'percent' ? 'bg-red-600 text-white shadow-lg' : 'text-zinc-500 hover:text-white'}`}
                                >
                                    Probability
                                </button>
                                <button
                                    onClick={() => setOddsFormat('american')}
                                    className={`px-3 py-1 text-[10px] font-bold uppercase tracking-widest transition-all rounded ${oddsFormat === 'american' ? 'bg-red-600 text-white shadow-lg' : 'text-zinc-500 hover:text-white'}`}
                                >
                                    American
                                </button>
                            </div>
                        </div>

                        <div className="text-red-600 font-bold uppercase tracking-widest text-[10px] sm:text-xs mb-1 flex items-center">
                            <span className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-red-600 rounded-full mr-2 animate-pulse"></span>
                            Next Event
                        </div>
                        <h2 className="text-2xl sm:text-3xl md:text-5xl font-black uppercase italic leading-tight text-white mb-3 drop-shadow-xl">
                            {featuredEvent.event_name}
                        </h2>

                        <div className="flex flex-wrap gap-3 sm:gap-4 text-zinc-300 font-medium text-[11px] sm:text-sm">
                            <div className="flex items-center gap-1.5 sm:gap-2">
                                <Calendar size={14} className="text-red-500" />
                                {featuredEvent.date}
                            </div>
                            <div className="flex items-center gap-1.5 sm:gap-2">
                                <MapPin size={14} className="text-red-500" />
                                {featuredEvent.location}
                            </div>
                        </div>
                    </div>
                </div>

                {/* FIGHT CARD LIST */}
                <div className="p-4 sm:p-8 w-full max-w-5xl mx-auto">
                    {/* Main Card */}
                    <div className="mb-6">
                        <div className="flex items-center gap-2 mb-4">
                            <span className="w-1 h-4 bg-red-600 rounded-full"></span>
                            <h3 className="text-zinc-400 font-black uppercase tracking-widest text-xs">Main Card</h3>
                        </div>
                        <div className="grid gap-3">
                            {featuredEvent.fights.filter(f => f.is_main_card !== false).map((fight, idx) => (
                                <FightCard key={idx} fight={fight} oddsFormat={oddsFormat} />
                            ))}
                        </div>
                    </div>

                    {/* Prelims - Only show if there are prelim fights */}
                    {featuredEvent.fights.some(f => f.is_main_card === false) && (
                        <div className="mb-6 pt-6 border-t border-zinc-900">
                            <div className="flex items-center gap-2 mb-4">
                                <span className="w-1 h-4 bg-zinc-600 rounded-full"></span>
                                <h3 className="text-zinc-500 font-black uppercase tracking-widest text-xs">Preliminary Card</h3>
                            </div>
                            <div className="grid gap-3">
                                {featuredEvent.fights.filter(f => f.is_main_card === false).map((fight, idx) => (
                                    <FightCard key={idx} fight={fight} oddsFormat={oddsFormat} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* RIGHT: Sidebar (Upcoming Schedule) */}
            <div className="lg:w-96 bg-zinc-950 border-l border-zinc-900 flex flex-col h-auto lg:h-screen lg:sticky lg:top-0">
                <div className="p-5 border-b border-zinc-900 bg-black/50 backdrop-blur-sm sticky top-0 z-20">
                    <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-500 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 bg-red-600 rounded-full"></span>
                        Fight Schedule
                    </h3>
                </div>

                <div className="flex lg:flex-col overflow-x-auto lg:overflow-y-auto flex-1 p-3 gap-3 scrollbar-none lg:scrollbar-thin scrollbar-thumb-zinc-800 scroll-smooth snap-x">
                    {upcomingEvents.map((event) => {
                        const isActive = featuredEvent.event_id === event.event_id;
                        const isNumberedEvent = /UFC\s+\d+/.test(event.event_name);

                        // Border Color Logic (Left Tab)
                        // Numbered: Gold (#BF953F), Fight Night: Red
                        const borderColorClass = isNumberedEvent ? "border-[#BF953F]" : "border-red-600";
                        const textColorClass = isNumberedEvent ? "text-[#BF953F]" : "text-red-600";
                        const ringColorClass = isNumberedEvent ? "ring-[#BF953F]" : "ring-red-600";

                        return (
                            <div
                                id={`event-${event.event_id}`}
                                key={event.event_id}
                                className={`group p-3 rounded-r transition-all cursor-pointer relative overflow-hidden shrink-0 w-[240px] lg:w-full snap-start 
                                    border-l-4 ${borderColorClass}
                                    ${isActive
                                        ? `bg-zinc-900 ring-2 ${ringColorClass} shadow-lg scale-[1.02] z-10`
                                        : 'bg-zinc-900/50 hover:bg-zinc-900 hover:pl-4 border-y border-r border-zinc-800 hover:border-zinc-700'
                                    }`}
                                onClick={() => {
                                    setFeaturedEvent(event);
                                    window.scrollTo({ top: 0, behavior: 'smooth' });
                                }}
                            >
                                <div className="flex justify-between items-start mb-1">
                                    <div className={`text-[10px] font-bold uppercase ${isActive ? textColorClass : 'text-zinc-500 group-hover:text-zinc-300'}`}>
                                        {event.date}
                                    </div>
                                </div>

                                <h4 className={`font-bold text-xs sm:text-sm leading-tight mb-2 truncate ${isActive ? 'text-white' : 'text-zinc-300 group-hover:text-white'}`}>
                                    {event.event_name}
                                </h4>

                                <div className="text-[10px] text-zinc-500 flex items-center gap-1 truncate">
                                    <MapPin size={10} /> {event.location}
                                </div>
                            </div>
                        );
                    })}

                    {upcomingEvents.length === 0 && (
                        <div className="text-zinc-600 text-center py-10 text-sm w-full">
                            No more upcoming events found.
                        </div>
                    )}
                </div>

                <div className="p-6 border-t border-zinc-900 bg-black text-center">
                    <div className="text-[10px] text-zinc-600 uppercase tracking-[0.3em] font-bold">
                        OCTAGON AI // v10.4
                    </div>
                </div>
            </div>
        </main>
    );
}
