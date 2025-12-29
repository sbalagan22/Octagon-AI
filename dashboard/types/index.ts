
export interface Prediction {
    winner: string;
    confidence: string;  // e.g. "85.0%"
    odds: {
        [key: string]: string;  // e.g. { "Fighter Name": "85.0%" }
    };
    factors?: {
        [key: string]: {
            slpm: number;
            kd_rate: number;
            td_rate: number;
            ctrl_rate: number;
            sub_rate: number;
            wins: number;
            losses: number;
            win_rate: number;
            height_cm: number;
            reach_cm: number;
            recent_form?: string;
        };
    };
    mov?: {
        [key: string]: {
            ko: string;
            sub: string;
            dec: string;
        };
    };
}

export interface ChartData {
    labels: string[];
    fighter_1_data: number[];
    fighter_2_data: number[];
}

export interface Fight {
    fighter_1: string;
    fighter_2: string;
    fighter_1_url: string;
    fighter_2_url: string;
    fighter_1_image?: string;
    fighter_2_image?: string;
    weight_class?: string;
    prediction?: Prediction;
    chart_data?: ChartData;
    prediction_status?: string;
    is_title_fight?: boolean;
    is_main_card?: boolean;
    market_odds?: {
        [key: string]: number | string;
    };
}

export interface Event {
    event_id: string;
    event_name: string;
    date: string;
    location: string;
    url: string;
    fights: Fight[];
}
