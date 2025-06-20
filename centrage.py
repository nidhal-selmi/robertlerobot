    # Centrage automatique

        image_center_x = frame.shape[1] // 2
        image_center_y = frame.shape[0] // 2
        delta_x = last_fraise_position[0] - image_center_x
        delta_y = last_fraise_position[1] - image_center_y

        seuil = 50

        if abs(delta_x) > seuil:
            if delta_x < 0:
                print("➡️ Bouger à droite pour centrer")
                send_command(f"MOVE {CENTER_STEPS_X:+d} 0")
            else:
                print("⬅️ Bouger à gauche pour centrer")
                send_command(f"MOVE {-CENTER_STEPS_X:+d} 0")
            time.sleep(0.2)
        elif abs(delta_y) > seuil:
            if delta_y < 0:
                print("⬇️ Bouger en bas pour centrer")
                send_command(f"MOVE 0 {CENTER_STEPS_Y:+d}")
            else:
                print("⬆️ Bouger en haut pour centrer")
                send_command(f"MOVE 0 {-CENTER_STEPS_Y:+d}")
            time.sleep(0.2)
        else:
            print("✅ Fraise centrée !")
            centrage_en_cours = False
