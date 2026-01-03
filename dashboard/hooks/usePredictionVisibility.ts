"use client";

import { useState, useEffect } from "react";

export function usePredictionVisibility() {
    const [showPredictions, setShowPredictions] = useState(true);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        const stored = localStorage.getItem("showPredictions");
        if (stored !== null) {
            setShowPredictions(JSON.parse(stored));
        }
        setMounted(true);
    }, []);

    const toggleVisibility = () => {
        const newValue = !showPredictions;
        setShowPredictions(newValue);
        localStorage.setItem("showPredictions", JSON.stringify(newValue));
    };

    return { showPredictions, toggleVisibility, mounted };
}
